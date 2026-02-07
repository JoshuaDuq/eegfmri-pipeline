package environment

import (
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/eeg-pipeline/tui/animation"
	"github.com/eeg-pipeline/tui/cloud"
	"github.com/eeg-pipeline/tui/components"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

type Environment int

const (
	EnvLocal Environment = iota
	EnvGoogleCloud
)

const (
	contentWidthPadding    = 6
	separatorWidthPadding  = 4
	maxSeparatorWidth      = 55
	minMainContentHeight   = 10
	headerHeightPadding    = 3
	footerHeightPadding    = 2
	optionNameWidth        = 20
	cloudPanelWidthPadding = 10
	footerWidthPadding     = 8
	maxErrorMsgLength      = 40
	helpOverlayWidth       = 45
	labelStyleWidth        = 15
)

const (
	vmStatusUnknown  = "unknown"
	vmStatusChecking = "checking"
	vmStatusStarting = "starting"
	vmStatusRunning  = "running"
	vmStatusStopped  = "stopped"
)

func (e Environment) String() string {
	switch e {
	case EnvLocal:
		return "Local"
	case EnvGoogleCloud:
		return "Google Cloud VM"
	}
	return "Unknown"
}

type envOption struct {
	env         Environment
	name        string
	description string
}

var environments = []envOption{
	{
		EnvLocal,
		"Local Machine",
		"Run pipelines on this computer",
	},
	{
		EnvGoogleCloud,
		"Google Cloud VM",
		"Process on remote VM",
	},
}

type Model struct {
	cursor   int
	Selected Environment
	Done     bool
	width    int
	height   int

	CloudConfig           cloud.Config
	VMStatus              string
	VMHostname            string
	VMError               error
	showCloudConfirmation bool
	animQueue             animation.Queue

	helpOverlay components.HelpOverlay
	showHelp    bool
}

func New() Model {
	help := components.NewHelpOverlay("Environment Shortcuts", helpOverlayWidth)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓", Description: "Select environment"},
	})
	help.AddSection("Actions", []components.HelpItem{
		{Key: "Enter", Description: "Confirm selection"},
		{Key: "?", Description: "Toggle help"},
	})
	help.AddSection("General", []components.HelpItem{
		{Key: "Q/Esc", Description: "Quit"},
	})

	m := Model{
		cursor:      0,
		Selected:    EnvLocal,
		CloudConfig: cloud.DefaultConfig(),
		VMStatus:    vmStatusUnknown,
		helpOverlay: help,
	}
	m.animQueue.Push(animation.ProgressPulseLoop())
	return m
}

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*styles.TickIntervalMs, func(t time.Time) tea.Msg {
		return tickMsg{}
	})
}

type tickMsg struct{}

func (m Model) isCloudSelected() bool {
	return m.cursor < len(environments) && environments[m.cursor].env == EnvGoogleCloud
}

func (m Model) moveCursorUp() Model {
	if m.cursor > 0 {
		m.cursor--
	} else {
		m.cursor = len(environments) - 1
	}
	return m
}

func (m Model) moveCursorDown() Model {
	if m.cursor < len(environments)-1 {
		m.cursor++
	} else {
		m.cursor = 0
	}
	return m
}

func (m Model) handleNavigation(key string) (Model, tea.Cmd, bool) {
	switch key {
	case "up", "k":
		m = m.moveCursorUp()
	case "down", "j":
		m = m.moveCursorDown()
	default:
		return m, nil, false
	}

	if m.showCloudConfirmation && !m.isCloudSelected() {
		m.showCloudConfirmation = false
	}

	if m.isCloudSelected() && m.VMStatus == vmStatusUnknown {
		m.VMStatus = vmStatusChecking
		return m, cloud.CheckVMStatus(m.CloudConfig), true
	}

	return m, nil, true
}

func (m Model) handleSelection(key string) (Model, tea.Cmd, bool) {
	switch key {
	case "enter", " ":
		if m.isCloudSelected() {
			if m.VMStatus == vmStatusRunning {
				m.Selected = EnvGoogleCloud
				m.Done = true
				return m, nil, true
			}
			m.showCloudConfirmation = true
			return m, nil, true
		}
		m.Selected = environments[m.cursor].env
		m.Done = true
		return m, nil, true
	}

	return m, nil, false
}

func (m Model) handleCloudConfirmation(key string) (Model, tea.Cmd, bool) {
	switch key {
	case "y", "Y":
		m.VMStatus = vmStatusStarting
		return m, cloud.StartVM(m.CloudConfig), true
	case "n", "N", "esc":
		m.showCloudConfirmation = false
		return m, nil, true
	}

	return m, nil, false
}

func (m Model) handleVMStatus(msg cloud.VMStatusMsg) Model {
	if msg.Error != nil {
		m.VMStatus = vmStatusStopped
		m.VMError = msg.Error
		m.showCloudConfirmation = false
	} else if msg.Running {
		m.VMStatus = vmStatusRunning
		m.VMHostname = msg.IP
		if m.showCloudConfirmation {
			m.showCloudConfirmation = false
			m.Selected = EnvGoogleCloud
			m.Done = true
		}
	} else {
		m.VMStatus = vmStatusStopped
	}
	return m
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tickMsg:
		m.animQueue.Tick()
		return m, m.tick()
	case tea.KeyMsg:
		key := msg.String()
		if m.showHelp {
			if key == "?" || key == "esc" {
				m.showHelp = false
				m.helpOverlay.Visible = false
				return m, nil
			}
			return m, nil
		}

		if key == "?" {
			m.showHelp = true
			m.helpOverlay.Visible = true
			return m, nil
		}

		if m.showCloudConfirmation {
			if updated, cmd, handled := m.handleCloudConfirmation(key); handled {
				return updated, cmd
			}
		}

		if updated, cmd, handled := m.handleNavigation(key); handled {
			return updated, cmd
		}

		if updated, cmd, handled := m.handleSelection(key); handled {
			return updated, cmd
		}

	case cloud.VMStatusMsg:
		m = m.handleVMStatus(msg)
		return m, nil

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
	}

	return m, nil
}

func (m Model) View() string {
	if m.showHelp {
		overlay := m.helpOverlay.View()
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, overlay)
	}

	separatorWidth := m.calculateSeparatorWidth(m.calculateContentWidth())

	header := m.renderWelcomeBanner() + "\n" + m.renderSystemInfo()
	headerHeight := strings.Count(header, "\n") + 2

	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + 2

	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < minMainContentHeight {
		mainHeight = minMainContentHeight
	}

	var mainContent strings.Builder
	mainContent.WriteString(m.renderTitle(separatorWidth))
	mainContent.WriteString("\n\n")
	mainContent.WriteString(m.renderOptions())

	if m.showCloudConfirmation {
		mainContent.WriteString("\n")
		mainContent.WriteString(m.renderCloudConfirmation())
	}

	if m.isCloudSelected() {
		mainContent.WriteString("\n")
		mainContent.WriteString(m.renderCloudPanel())
	}

	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent.String())

	return header + "\n" + mainContentStyled + "\n" + footer
}

func (m Model) calculateContentWidth() int {
	contentWidth := m.width - contentWidthPadding
	if contentWidth < styles.MinContentWidth {
		contentWidth = styles.MinContentWidth
	}
	if contentWidth > styles.MaxContentWidth {
		contentWidth = styles.MaxContentWidth
	}
	return contentWidth
}

func (m Model) calculateSeparatorWidth(contentWidth int) int {
	separatorWidth := contentWidth - separatorWidthPadding
	if separatorWidth > maxSeparatorWidth {
		separatorWidth = maxSeparatorWidth
	}
	return separatorWidth
}

func (m Model) renderWelcomeBanner() string {
	logo := lipgloss.NewStyle().
		Bold(true).
		Foreground(styles.Primary).
		Render("EEG Pipeline")

	version := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render("  v1.0")

	lineWidth := m.width - 4
	if lineWidth < 0 {
		lineWidth = 0
	}

	return "  " + logo + version + "\n" + styles.RenderHeaderSeparator(lineWidth)
}

func isRunningOnGCP() bool {
	client := &http.Client{Timeout: 1 * time.Second}
	req, err := http.NewRequest("GET", "http://metadata.google.internal/computeMetadata/v1/instance/name", nil)
	if err != nil {
		return false
	}
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (m Model) renderSystemInfo() string {
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}

	cpuCount := runtime.NumCPU()
	osArch := fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)

	envType := "Local"
	if isRunningOnGCP() {
		envType = "Virtual Machine"
	}

	sep := lipgloss.NewStyle().Foreground(styles.Border).Render("  |  ")
	parts := []string{
		styles.RenderKeyValue("Env", envType, 5),
		styles.RenderKeyValue("Host", hostname, 6),
		styles.RenderKeyValue("OS", osArch, 4),
		styles.RenderKeyValue("CPUs", fmt.Sprintf("%d", cpuCount), 6),
	}
	return "  " + strings.Join(parts, sep)
}

func (m Model) renderTitle(separatorWidth int) string {
	title := styles.RenderSectionLabel("Select Environment")
	subtitle := lipgloss.NewStyle().Foreground(styles.TextDim).Render("  where should pipelines run?")
	return title + subtitle + "\n" + styles.RenderDivider(separatorWidth)
}

func (m Model) renderOptions() string {
	var builder strings.Builder

	for i, opt := range environments {
		isFocused := i == m.cursor
		cursor := "  "
		marker := styles.RenderRadio(false, false)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text).Width(optionNameWidth)
		descStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

		if isFocused {
			cursor = styles.RenderCursor()
			marker = styles.RenderRadio(true, true)
			nameStyle = nameStyle.Bold(true).Foreground(styles.Primary)
		}

		line := cursor + marker + " " + nameStyle.Render(opt.name) + descStyle.Render(opt.description)

		if opt.env == EnvGoogleCloud {
			line += "  " + m.renderVMStatusBadge()
		}

		builder.WriteString(line + "\n")
	}

	return builder.String()
}

func (m Model) renderCloudConfirmation() string {
	messageStyle := lipgloss.NewStyle().
		Foreground(styles.TextDim).
		Italic(true).
		PaddingLeft(2)

	hintStyle := lipgloss.NewStyle().
		Foreground(styles.Muted).
		PaddingLeft(2)

	message := messageStyle.Render("Are you sure you want to start the cloud environment?")
	hint := hintStyle.Render("(Y)es / (N)o")

	return message + "\n" + hint
}

func (m Model) renderVMStatusBadge() string {
	switch m.VMStatus {
	case vmStatusChecking, vmStatusStarting:
		label := vmStatusChecking
		if m.VMStatus == vmStatusStarting {
			label = vmStatusStarting
		}
		kind, progress := m.animQueue.Current()
		icon := styles.ActiveMark
		if kind == animation.KindProgressPulse && progress >= 0.5 {
			icon = styles.PendingMark
		}
		return lipgloss.NewStyle().Foreground(styles.Accent).Render("[" + icon + " " + label + "]")
	case vmStatusRunning:
		return lipgloss.NewStyle().Foreground(styles.Success).Render("[" + styles.ActiveMark + " running]")
	case vmStatusStopped:
		return lipgloss.NewStyle().Foreground(styles.Warning).Render("[" + styles.PendingMark + " stopped]")
	default:
		return lipgloss.NewStyle().Foreground(styles.Muted).Render("[? unknown]")
	}
}

func (m Model) renderCloudPanel() string {
	var content strings.Builder

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(labelStyleWidth)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	content.WriteString(labelStyle.Render("SSH host:") + valueStyle.Render(m.CloudConfig.RemoteHost) + "\n")
	content.WriteString(labelStyle.Render("Remote path:") + valueStyle.Render(m.CloudConfig.RemoteBase) + "\n")
	content.WriteString(labelStyle.Render("GCP instance:") + valueStyle.Render(m.CloudConfig.GCPInstance) + "\n")

	statusLabel := labelStyle.Render("Status:")
	statusLine := m.renderCloudStatusLine(statusLabel, valueStyle, labelStyle)
	content.WriteString(statusLine)

	return styles.PanelStyle.Width(m.width - cloudPanelWidthPadding).Render(content.String())
}

func (m Model) renderCloudStatusLine(statusLabel string, valueStyle, labelStyle lipgloss.Style) string {
	switch m.VMStatus {
	case vmStatusRunning:
		statusText := lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render(styles.ActiveMark + " Connected")
		if m.VMHostname != "" {
			statusText += " (" + valueStyle.Render(m.VMHostname) + ")"
		}
		return statusLabel + statusText + "\n"

	case vmStatusStopped:
		if m.VMError != nil {
			statusText := lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark + " Connection failed")
			result := statusLabel + statusText + "\n"
			errMsg := m.VMError.Error()
			if len(errMsg) > maxErrorMsgLength {
				errMsg = errMsg[:maxErrorMsgLength] + "..."
			}
			result += labelStyle.Render("") + lipgloss.NewStyle().Foreground(styles.TextDim).Render(errMsg) + "\n"
			return result
		}
		statusText := lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.PendingMark + " Stopped")
		return statusLabel + statusText + "\n"

	case vmStatusChecking:
		statusText := lipgloss.NewStyle().Foreground(styles.Accent).Render("checking connection...")
		return statusLabel + statusText + "\n"

	case vmStatusStarting:
		statusText := lipgloss.NewStyle().Foreground(styles.Accent).Render("starting VM...")
		return statusLabel + statusText + "\n"

	default:
		statusText := lipgloss.NewStyle().Foreground(styles.Muted).Render("unknown")
		return statusLabel + statusText + "\n"
	}
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("\u2191\u2193", "Navigate"),
		styles.RenderKeyHint("Enter", "Select"),
	}

	if m.showCloudConfirmation {
		hints = append(hints,
			styles.RenderKeyHint("Y", "Yes"),
			styles.RenderKeyHint("N", "No"),
		)
	}

	hints = append(hints, styles.RenderKeyHint("Q", "Quit"))

	width := m.width - footerWidthPadding
	if width < 20 {
		width = 20
	}
	divider := styles.RenderDivider(width)
	bar := styles.FooterStyle.Width(width).Render(strings.Join(hints, styles.RenderFooterSeparator()))
	return divider + "\n" + bar
}
