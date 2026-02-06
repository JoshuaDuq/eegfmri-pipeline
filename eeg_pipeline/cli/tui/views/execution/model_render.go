package execution

import (
	"fmt"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/styles"

	"github.com/charmbracelet/lipgloss"
)

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

// renderCompletionSummary produces a compact summary card shown
// once execution has finished, including timing, errors and outputs.
func (m Model) renderCompletionSummary() string {
	var b strings.Builder

	var icon, statusText string
	var statusColor lipgloss.Color
	var borderColor lipgloss.Color

	switch m.Status {
	case StatusSuccess:
		icon = styles.CheckMark
		statusText = "Completed successfully"
		statusColor = styles.Success
		borderColor = styles.Success
	case StatusFailed:
		icon = styles.CrossMark
		statusText = "Execution failed"
		statusColor = styles.Error
		borderColor = styles.Error
	case StatusCancelled:
		icon = "⊘"
		statusText = "Execution cancelled"
		statusColor = styles.Warning
		borderColor = styles.Warning
	default:
		return ""
	}

	// Status line with icon
	statusLine := lipgloss.NewStyle().
		Bold(true).
		Foreground(statusColor).
		Render(icon + "  " + statusText)
	b.WriteString(statusLine + "\n")

	// Divider line
	divWidth := 40
	div := lipgloss.NewStyle().Foreground(statusColor).Render(strings.Repeat("─", divWidth))
	b.WriteString(div + "\n\n")

	// Stats in a clean layout
	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(14)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	// Duration
	duration := m.getDuration()
	durationStr := formatDuration(duration)
	b.WriteString(labelStyle.Render("duration:") + valueStyle.Render(durationStr) + "\n")

	// Subjects processed
	if m.SubjectTotal > 0 && m.SubjectCurrent > 0 {
		subjProgress := float64(m.SubjectCurrent) / float64(m.SubjectTotal)
		subjInfo := fmt.Sprintf("%d of %d (%.0f%%)", m.SubjectCurrent, m.SubjectTotal, subjProgress*100)
		b.WriteString(labelStyle.Render("subjects:") + valueStyle.Render(subjInfo) + "\n")
	}

	// Exit code for failures
	if m.Status == StatusFailed {
		exitStyle := lipgloss.NewStyle().Foreground(styles.Error).Bold(true)
		b.WriteString(labelStyle.Render("Exit code:") + exitStyle.Render(fmt.Sprintf("%d", m.ExitCode)) + "\n")
	}

	// Error count with visual indicator
	if len(m.ErrorLines) > 0 {
		errStyle := lipgloss.NewStyle().Foreground(styles.Warning)
		errText := fmt.Sprintf("%d detected", len(m.ErrorLines))
		if len(m.ErrorLines) > 5 {
			errText += " (scroll log to view)"
		}
		b.WriteString(labelStyle.Render("errors:") + errStyle.Render(errText) + "\n")
	}

	// Log lines processed
	logCount := fmt.Sprintf("%d lines", len(m.OutputLines))
	if m.LogTruncated {
		logCount += lipgloss.NewStyle().Foreground(styles.Warning).Render(" (truncated)")
	}
	b.WriteString(labelStyle.Render("log:") + valueStyle.Render(logCount) + "\n")

	// Show output paths for successful runs
	if m.Status == StatusSuccess && m.RepoRoot != "" {
		b.WriteString("\n")
		outputHeader := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render("Output locations:")
		b.WriteString(outputHeader + "\n")

		outputPaths := m.GetOutputPaths()
		for i, p := range outputPaths {
			pathStyle := lipgloss.NewStyle().Foreground(styles.Text)
			numStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
			b.WriteString(numStyle.Render(fmt.Sprintf("[%d]", i+1)) + " " + pathStyle.Render(p) + "\n")
		}
	}

	// Add quick action buttons
	b.WriteString("\n")
	switch m.Status {
	case StatusSuccess:
		// Styled action buttons for success with results browsing
		enterBtn := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Success).
			Bold(true).
			Padding(0, 1).
			Render("[Enter] Menu")
		openBtn := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1).
			Render("[O] Open Results")
		copyBtn := lipgloss.NewStyle().
			Foreground(styles.Text).
			Background(styles.Secondary).
			Padding(0, 1).
			Render("[C] Copy Log")
		b.WriteString(enterBtn + "  " + openBtn + "  " + copyBtn)
	case StatusFailed:
		// Prominent action buttons for failure - recovery options highlighted
		retryBtn := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Warning).
			Bold(true).
			Padding(0, 1).
			Render("[R] Retry")
		copyBtn := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#FFFFFF")).
			Background(styles.Error).
			Bold(true).
			Padding(0, 1).
			Render("[C] Copy Log")
		menuBtn := lipgloss.NewStyle().
			Foreground(styles.Text).
			Background(styles.Secondary).
			Padding(0, 1).
			Render("[Enter] Menu")
		b.WriteString(retryBtn + "  " + copyBtn + "  " + menuBtn)
	case StatusCancelled:
		retryBtn := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Bold(true).
			Padding(0, 1).
			Render("[R] Retry")
		menuBtn := lipgloss.NewStyle().
			Foreground(styles.Text).
			Background(styles.Secondary).
			Padding(0, 1).
			Render("[Enter] Menu")
		b.WriteString(retryBtn + "  " + menuBtn)
	}

	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(borderColor).
		Padding(1, 2).
		Width(m.panelWidth())

	return cardStyle.Render(b.String())
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
	title := "Pipeline execution"
	if m.IsCloud {
		title = "Cloud execution"
	}

	borderColor := styles.Secondary
	switch m.Status {
	case StatusRunning:
		borderColor = styles.Accent
	case StatusSuccess:
		borderColor = styles.Success
	case StatusFailed:
		borderColor = styles.Error
	case StatusCancelled:
		borderColor = styles.Warning
	}

	header := lipgloss.NewStyle().
		Bold(true).
		Foreground(borderColor).
		Render(title)

	return lipgloss.PlaceHorizontal(m.width-4, lipgloss.Center, header)
}

// renderInfoPanel renders high‑level execution metadata such as
// elapsed time, current subject and failure counts.
func (m Model) renderInfoPanel() string {
	info := strings.Builder{}

	info.WriteString(styles.SectionTitleStyle.Render("Summary") + "\n")

	if m.StartTime.Unix() > 0 {
		duration := m.getDuration()
		timeLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("elapsed:")
		timeValue := lipgloss.NewStyle().Foreground(styles.Text).Render(formatDuration(duration))
		info.WriteString(timeLabel + timeValue)

		if m.Status == StatusRunning && m.EstimatedRemaining > 0 {
			etaLabel := lipgloss.NewStyle().Foreground(styles.TextDim).MarginLeft(2).Render("  ETA:")
			etaValue := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(formatDuration(m.EstimatedRemaining))
			info.WriteString(etaLabel + etaValue)
		}
		info.WriteString("\n")
	}

	if m.SubjectTotal > 0 && m.CurrentSubject != "" && m.Status == StatusRunning {
		subjLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("subject:")
		subjValue := lipgloss.NewStyle().Foreground(styles.Primary).Bold(true).Render(m.CurrentSubject)
		subjProgress := lipgloss.NewStyle().Foreground(styles.TextDim).Render(
			fmt.Sprintf(" (%d of %d)", m.SubjectCurrent, m.SubjectTotal))
		info.WriteString(subjLabel + subjValue + subjProgress)

		if len(m.SubjectDurations) > 0 {
			avgDuration := m.averageSubjectDuration()
			avgLabel := lipgloss.NewStyle().Foreground(styles.TextDim).MarginLeft(2).Render("  Avg:")
			avgValue := lipgloss.NewStyle().Foreground(styles.Text).Render(formatDuration(avgDuration))
			info.WriteString(avgLabel + avgValue)
		}
		info.WriteString("\n")
	}

	if len(m.FailedSubjects) > 0 {
		failLabel := lipgloss.NewStyle().Foreground(styles.Error).Width(10).Render("failed:")
		failValue := lipgloss.NewStyle().Foreground(styles.Error).Bold(true).Render(
			fmt.Sprintf("%d subject(s)", len(m.FailedSubjects)))
		info.WriteString(failLabel + failValue + "\n")
	}

	return m.renderSidebarCard(info.String())
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
// overall completion, current subject/step and optional cloud stages.
func (m Model) renderProgressSection() string {
	var b strings.Builder

	// Responsive Layout Decision
	contentWidth := m.contentWidth()
	isNarrow := contentWidth < 100
	isShort := m.height < 30

	// Section header with status icon
	var progressIcon string
	var iconStyle lipgloss.Style
	switch m.Status {
	case StatusSuccess:
		progressIcon = styles.CheckMark
		iconStyle = lipgloss.NewStyle().Foreground(styles.Success)
	case StatusFailed:
		progressIcon = styles.CrossMark
		iconStyle = lipgloss.NewStyle().Foreground(styles.Error)
	default:
		progressIcon = styles.ActiveMark
		iconStyle = lipgloss.NewStyle().Foreground(styles.Accent)
	}

	b.WriteString(iconStyle.Render(progressIcon) + " " + styles.SectionTitleStyle.Render("Progress") + "\n")

	// Overall progress bar
	progressLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("overall")
	barWidth := contentWidth - 22
	if isNarrow {
		barWidth = contentWidth - 15
	}
	b.WriteString("  " + progressLabel + m.renderAnimatedProgressBar(m.Progress, barWidth) + "\n")

	// Metrics Dashboard (Responsive)
	if !isShort {
		b.WriteString("\n" + m.renderMetricsDashboard() + "\n\n")
	}

	// Subject counter with visual indicator
	if m.SubjectTotal > 0 && m.SubjectCurrent > 0 {
		subjectProgress := float64(m.SubjectCurrent) / float64(m.SubjectTotal)

		// Create a mini visual representation (hide on narrow screens)
		var subjectIcons string
		if !isNarrow {
			for i := 0; i < m.SubjectTotal && i < 10; i++ {
				if i < m.SubjectCurrent {
					subjectIcons += lipgloss.NewStyle().Foreground(styles.Success).Render("●")
				} else if i == m.SubjectCurrent {
					subjectIcons += lipgloss.NewStyle().Foreground(styles.Accent).Render("○")
				} else {
					subjectIcons += lipgloss.NewStyle().Foreground(styles.Secondary).Render("○")
				}
			}
			if m.SubjectTotal > 10 {
				subjectIcons += lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf(" +%d", m.SubjectTotal-10))
			}
		}

		subjectText := fmt.Sprintf("Subject %d/%d  ", m.SubjectCurrent, m.SubjectTotal)
		b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).PaddingLeft(12).Render(subjectText))
		if !isNarrow {
			b.WriteString(subjectIcons)
		}
		b.WriteString(lipgloss.NewStyle().Foreground(styles.Muted).Render(fmt.Sprintf("  %.0f%%", subjectProgress*100)) + "\n")
	}

	// Current step progress with enhanced styling
	if m.CurrentSubject != "" || m.CurrentOperation != "" {
		b.WriteString("\n")
		stepLabel := lipgloss.NewStyle().Foreground(styles.TextDim).Width(10).Render("step")

		if m.OperationTotal > 0 {
			stepProgress := float64(m.OperationCurrent) / float64(m.OperationTotal)
			b.WriteString("  " + stepLabel + m.renderMiniProgressBar(stepProgress, contentWidth-32))

			// Operation name with step counter
			opText := fmt.Sprintf(" %s (%d/%d)", m.CurrentOperation, m.OperationCurrent, m.OperationTotal)
			b.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(opText) + "\n")
		} else {
			// Just show subject and operation
			b.WriteString("  " + stepLabel)
			b.WriteString(lipgloss.NewStyle().Foreground(styles.Accent).Bold(true).Render(m.CurrentSubject))
			if m.CurrentOperation != "" {
				b.WriteString(" " + lipgloss.NewStyle().Foreground(styles.TextDim).Render("→ "+m.CurrentOperation))
			}
			b.WriteString("\n")
		}
	}

	// Cloud stages (if cloud mode)
	if m.IsCloud {
		b.WriteString("\n  " + m.renderCloudStages() + "\n\n")
	}

	// Status Badge with enhanced styling
	b.WriteString("\n  " + m.renderStatus() + "\n")

	return b.String()
}

// renderMetricsDashboard renders memory usage, optional epoch info
// and per‑core CPU utilization when resource monitoring is active.
func (m Model) renderMetricsDashboard() string {
	if m.Status == StatusPending {
		return ""
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	metricBox := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(1, 2).
		MarginRight(1)

	// Memory Metric
	mem := fmt.Sprintf("%.1f GB", m.MemoryUsage)
	memView := metricBox.Render(labelStyle.Render("MEM: ") + valueStyle.Render(mem))

	// Epoch/Iteration Info
	epochView := ""
	if m.EpochInfo != "" {
		epochView = metricBox.Render(labelStyle.Render("ITEM: ") + valueStyle.Render(m.EpochInfo))
	}

	// Build top row with memory and epoch
	topRow := lipgloss.JoinHorizontal(lipgloss.Top, memView, epochView)

	// Per-core CPU display
	cpuCoresView := m.renderPerCoreCPU()

	return topRow + "\n\n" + cpuCoresView
}

// renderPerCoreCPU renders a visual display of per-core CPU usage.
// Uses fixed-width cells to avoid wrapping and keep rows aligned.
func (m Model) renderPerCoreCPU() string {
	if m.NumCPUCores == 0 || len(m.CPUCoreUsages) == 0 {
		// Fallback to simple display if no per-core data
		labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
		valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
		return labelStyle.Render("  CPU: ") + valueStyle.Render(fmt.Sprintf("%.1f%%", m.CPUUsage))
	}

	var b strings.Builder

	numCores := len(m.CPUCoreUsages)
	availableWidth := m.contentWidth() - 2 // indent
	const (
		minCellWidth = 12
		maxCellWidth = 18
		minBarWidth  = 6
	)
	if availableWidth < minCellWidth {
		availableWidth = minCellWidth
	}

	coresPerRow := availableWidth / minCellWidth
	if coresPerRow < 1 {
		coresPerRow = 1
	}
	if coresPerRow > numCores {
		coresPerRow = numCores
	}

	slotWidth := availableWidth / coresPerRow
	if slotWidth > maxCellWidth {
		slotWidth = maxCellWidth
	}

	labelStyle := lipgloss.NewStyle().Foreground(styles.Muted)
	pctStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	rowStyle := lipgloss.NewStyle().PaddingLeft(2)

	// Render cores in rows using fixed-width cells
	for row := 0; row*coresPerRow < numCores; row++ {
		start := row * coresPerRow
		end := start + coresPerRow
		if end > numCores {
			end = numCores
		}

		var cells []string
		for i := start; i < end; i++ {
			coreUsage := m.CPUCoreUsages[i]
			if coreUsage > 100 {
				coreUsage = 100
			}
			if coreUsage < 0 {
				coreUsage = 0
			}

			label := labelStyle.Render(fmt.Sprintf("C%-2d", i))
			pct := pctStyle.Render(fmt.Sprintf("%3.0f%%", coreUsage))
			line1 := label + " " + pct

			barWidth := slotWidth - 2
			if barWidth < minBarWidth {
				barWidth = minBarWidth
			}
			bar := m.renderCoreMiniBar(coreUsage, barWidth)
			line2 := " " + bar

			cell := lipgloss.NewStyle().Width(slotWidth).Height(2).Render(line1 + "\n" + line2)
			cells = append(cells, cell)
		}

		rowLine := rowStyle.Render(lipgloss.JoinHorizontal(lipgloss.Top, cells...))
		b.WriteString(rowLine)
		if end < numCores {
			b.WriteString("\n")
		}
	}

	// Overall CPU summary
	b.WriteString("\n\n")
	totalStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Accent).Bold(true)
	avgUsage := 0.0
	for _, usage := range m.CPUCoreUsages {
		avgUsage += usage
	}
	if numCores > 0 {
		avgUsage /= float64(numCores)
	}
	b.WriteString("  " + totalStyle.Render("Total: ") + valueStyle.Render(fmt.Sprintf("%.1f%%", m.CPUUsage)))
	b.WriteString(totalStyle.Render("  Avg/Core: ") + valueStyle.Render(fmt.Sprintf("%.1f%%", avgUsage)))
	b.WriteString(totalStyle.Render(fmt.Sprintf("  (%d cores)", numCores)))

	return b.String()
}

// renderCoreMiniBar renders a tiny progress bar for a single CPU core
func (m Model) renderCoreMiniBar(usage float64, width int) string {
	filled := int(usage / 100.0 * float64(width))
	if filled < 0 {
		filled = 0
	}
	if filled > width {
		filled = width
	}

	// Color based on usage level
	var barColor lipgloss.Color
	if usage < 30 {
		barColor = styles.Success // Green for low usage
	} else if usage < 70 {
		barColor = styles.Accent // Teal for medium usage
	} else if usage < 90 {
		barColor = styles.Warning // Orange for high usage
	} else {
		barColor = styles.Error // Red for very high usage
	}

	filledStyle := lipgloss.NewStyle().Foreground(barColor)
	emptyStyle := lipgloss.NewStyle().Foreground(styles.Secondary)

	bar := filledStyle.Render(strings.Repeat("█", filled))
	bar += emptyStyle.Render(strings.Repeat("░", width-filled))

	return bar
}

func (m Model) renderLogSection() string {
	var b strings.Builder
	contentWidth := m.logViewport.Width

	// Copy mode banner
	if m.copyMode {
		copyBanner := lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#000000")).
			Background(styles.Accent).
			Padding(0, 2).
			Render("Copy mode: Select text with mouse, then Cmd+C. Press M or Esc to exit.")
		b.WriteString(lipgloss.NewStyle().Width(contentWidth).Render(copyBanner) + "\n")
	}

	logHeader := styles.SectionTitleStyle.Render("Log")
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

	fillStyle := lipgloss.NewStyle().Foreground(styles.Accent)
	emptyStyle := lipgloss.NewStyle().Foreground(styles.Secondary)

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

// renderMiniProgressBar renders a lightweight, single‑color bar for
// step‑level progress where space is more constrained.
func (m Model) renderMiniProgressBar(p float64, width int) string {
	if width < styles.MinProgressBarWidth {
		width = styles.MinProgressBarWidth
	}

	filled := int(p * float64(width))
	bar := lipgloss.NewStyle().Foreground(styles.Accent).Render(strings.Repeat("━", filled))
	empty := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", width-filled))
	return bar + empty
}

// renderStatus renders a small status badge summarizing the current
// execution state with an appropriate color and icon.
func (m Model) renderStatus() string {
	style := lipgloss.NewStyle().Bold(true).Padding(0, 1)
	switch m.Status {
	case StatusRunning:
		return style.Background(styles.Accent).Foreground(lipgloss.Color("#000000")).Render(styles.ActiveMark + " Running ")
	case StatusSuccess:
		return style.Background(styles.Success).Foreground(lipgloss.Color("#000000")).Render(styles.CheckMark + " Success ")
	case StatusFailed:
		return style.Background(styles.Error).Foreground(lipgloss.Color("#FFFFFF")).Render(styles.CrossMark + " Failed ")
	case StatusCancelled:
		return style.Background(styles.Muted).Foreground(lipgloss.Color("#FFFFFF")).Render(" Cancelled ")
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
		// Footer for copy mode
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
			styles.RenderKeyHint("↑/↓", "Scroll"),
			styles.RenderKeyHint("G/Shift+G", "Top/Bottom"),
		}
	}

	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, styles.RenderFooterSeparator()))
}

///////////////////////////////////////////////////////////////////
// Resource Monitoring
///////////////////////////////////////////////////////////////////

// startResourceMonitoring begins monitoring CPU and memory usage of the process
