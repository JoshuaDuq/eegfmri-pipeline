package environment

import (
	"fmt"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

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
	tickInterval = 150 * time.Millisecond

	contentWidthPadding     = 6
	separatorWidthPadding   = 4
	maxSeparatorWidth       = 55
	minMainContentHeight    = 10
	headerHeightPadding     = 3
	footerHeightPadding     = 2
	optionNameWidth         = 20
	cloudPanelWidthPadding  = 10
	footerWidthPadding      = 8
	maxErrorMsgLength       = 40
	helpOverlayWidth        = 45
	labelStyleWidth         = 15
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

	CloudConfig        cloud.Config
	VMStatus           string
	VMHostname         string
	VMError            error
	showCloudConfirmation bool

	helpOverlay components.HelpOverlay
	showHelp    bool

	ticker int
}

type TickMsg struct{}

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

	return Model{
		cursor:      0,
		Selected:    EnvLocal,
		CloudConfig: cloud.DefaultConfig(),
		VMStatus:    vmStatusUnknown,
		helpOverlay: help,
	}
}

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(tickInterval, func(t time.Time) tea.Msg {
		return TickMsg{}
	})
}

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
	case TickMsg:
		m.ticker++
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

	contentWidth := m.calculateContentWidth()
	separatorWidth := m.calculateSeparatorWidth(contentWidth)

	header := m.renderWelcomeBanner() + "\n\n" + m.renderSystemInfo()
	headerHeight := strings.Count(header, "\n") + headerHeightPadding

	footer := m.renderFooter()
	footerHeight := strings.Count(footer, "\n") + footerHeightPadding

	mainHeight := m.height - headerHeight - footerHeight
	if mainHeight < minMainContentHeight {
		mainHeight = minMainContentHeight
	}

	var mainContent strings.Builder
	mainContent.WriteString(m.renderTitle(separatorWidth))
	mainContent.WriteString("\n\n")
	mainContent.WriteString(m.renderOptions())
	mainContent.WriteString("\n")

	if m.showCloudConfirmation {
		mainContent.WriteString(m.renderCloudConfirmation())
		mainContent.WriteString("\n")
	}

	if m.isCloudSelected() {
		mainContent.WriteString(m.renderCloudPanel())
		mainContent.WriteString("\n")
	}

	mainContentStyled := lipgloss.NewStyle().
		Height(mainHeight).
		Render(mainContent.String())

	return header + "\n\n" + mainContentStyled + "\n" + footer
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
		Render("◆ EEG PIPELINE")

	version := lipgloss.NewStyle().
		Foreground(styles.Muted).
		Render(" v1.0")

	return logo + version
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
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}

	cpuCount := runtime.NumCPU()
	osArch := fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)

	var info []string

	envType := "Local"
	if isRunningOnGCP() {
		envType = "Virtual Machine"
	}
	info = append(info, infoStyle.Render("Environment: ")+valueStyle.Render(envType))
	info = append(info, infoStyle.Render("Host: ")+valueStyle.Render(hostname))
	info = append(info, infoStyle.Render("OS: ")+valueStyle.Render(osArch))
	info = append(info, infoStyle.Render("CPUs: ")+valueStyle.Render(fmt.Sprintf("%d", cpuCount)))

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	return strings.Join(info, separator)
}

func (m Model) renderTitle(separatorWidth int) string {
	titleStyle := lipgloss.NewStyle().Bold(true).Foreground(styles.Primary)
	subtitleStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

	title := titleStyle.Render("Select Environment")
	subtitle := subtitleStyle.Render("  Where should pipelines run?")
	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render(strings.Repeat("─", separatorWidth))

	return title + subtitle + "\n" + separator
}

func (m Model) renderOptions() string {
	var builder strings.Builder

	for i, opt := range environments {
		isFocused := i == m.cursor
		cursor := "  "
		marker := styles.RenderRadio(false, false)
		nameStyle := lipgloss.NewStyle().Foreground(styles.Text)
		descStyle := lipgloss.NewStyle().Foreground(styles.TextDim)

		if isFocused {
			cursor = styles.SelectedMark + " "
			marker = styles.RenderRadio(true, true)
			nameStyle = nameStyle.Bold(true).Foreground(styles.Primary)
		}

		paddingNeeded := optionNameWidth - len(opt.name)
		paddedName := opt.name + strings.Repeat(" ", paddingNeeded)

		line := cursor + marker + " " + nameStyle.Render(paddedName) + descStyle.Render(opt.description)

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

	message := messageStyle.Render("Are you sure you want to start the Cloud environment?")
	hint := hintStyle.Render("(Y)es / (N)o")

	return message + "\n" + hint
}

func (m Model) renderVMStatusBadge() string {
	switch m.VMStatus {
	case vmStatusChecking, vmStatusStarting:
		animationFrames := []string{"◐", "◓", "◑", "◒"}
		frameIndex := m.ticker % len(animationFrames)
		currentFrame := animationFrames[frameIndex]
		label := vmStatusChecking
		if m.VMStatus == vmStatusStarting {
			label = vmStatusStarting
		}
		return lipgloss.NewStyle().Foreground(styles.Accent).Render("[" + currentFrame + " " + label + "]")
	case vmStatusRunning:
		return lipgloss.NewStyle().Foreground(styles.Success).Render("[" + styles.ActiveMark + " running]")
	case vmStatusStopped:
		return lipgloss.NewStyle().Foreground(styles.Warning).Render("[" + styles.PendingMark + " stopped]")
	default:
		return lipgloss.NewStyle().Foreground(styles.Muted).Render("[? unknown]")
	}
}

func (m Model) renderCloudPanel() string {
	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(0, 1).
		Width(m.width - cloudPanelWidthPadding)

	var content strings.Builder

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(labelStyleWidth)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	content.WriteString(labelStyle.Render("SSH Host:") + valueStyle.Render(m.CloudConfig.RemoteHost) + "\n")
	content.WriteString(labelStyle.Render("Remote Path:") + valueStyle.Render(m.CloudConfig.RemoteBase) + "\n")
	content.WriteString(labelStyle.Render("GCP Instance:") + valueStyle.Render(m.CloudConfig.GCPInstance) + "\n")

	statusLabel := labelStyle.Render("Status:")
	statusLine := m.renderCloudStatusLine(statusLabel, valueStyle, labelStyle)
	content.WriteString(statusLine)

	return boxStyle.Render(content.String())
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
		statusText := lipgloss.NewStyle().Foreground(styles.Accent).Render("Checking connection...")
		return statusLabel + statusText + "\n"

	case vmStatusStarting:
		statusText := lipgloss.NewStyle().Foreground(styles.Accent).Render("Starting VM...")
		return statusLabel + statusText + "\n"

	default:
		statusText := lipgloss.NewStyle().Foreground(styles.Muted).Render("Unknown")
		return statusLabel + statusText + "\n"
	}
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Select"),
	}

	if m.showCloudConfirmation {
		hints = append(hints,
			styles.RenderKeyHint("Y", "Yes"),
			styles.RenderKeyHint("N", "No"),
		)
	}

	hints = append(hints, styles.RenderKeyHint("Q", "Quit"))

	separator := "  "
	return styles.FooterStyle.Width(m.width - footerWidthPadding).Render(strings.Join(hints, separator))
}
