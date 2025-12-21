package environment

import (
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

///////////////////////////////////////////////////////////////////
// Types
///////////////////////////////////////////////////////////////////

type Environment int

const (
	EnvLocal Environment = iota
	EnvGoogleCloud
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
	icon        string
	pros        []string
}

var environments = []envOption{
	{
		EnvLocal,
		"Local Machine",
		"Run pipelines on this computer",
		"▸",
		[]string{"No network required", "Full resource access", "Immediate start"},
	},
	{
		EnvGoogleCloud,
		"Google Cloud VM",
		"Process on remote GPU-enabled VM",
		"▸",
		[]string{"More compute power", "GPU acceleration", "Persistent storage"},
	},
}

///////////////////////////////////////////////////////////////////
// Model
///////////////////////////////////////////////////////////////////

type Model struct {
	cursor   int
	Selected Environment
	Done     bool
	width    int
	height   int

	CloudConfig cloud.Config
	VMStatus    string
	VMHostname  string
	VMError     error

	// Help overlay
	helpOverlay components.HelpOverlay
	showHelp    bool

	ticker int
}

type TickMsg struct{}

///////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////

func New() Model {
	help := components.NewHelpOverlay("Environment Shortcuts", 45)
	help.AddSection("Navigation", []components.HelpItem{
		{Key: "↑/↓", Description: "Select environment"},
		{Key: "L", Description: "Choose Local"},
		{Key: "G", Description: "Choose Cloud"},
	})
	help.AddSection("Cloud", []components.HelpItem{
		{Key: "C", Description: "Check VM status"},
		{Key: "S", Description: "Start VM"},
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
		VMStatus:    "unknown",
		helpOverlay: help,
	}
}

///////////////////////////////////////////////////////////////////
// Tea Model Implementation
///////////////////////////////////////////////////////////////////

func (m Model) Init() tea.Cmd {
	return m.tick()
}

func (m Model) tick() tea.Cmd {
	return tea.Tick(time.Millisecond*150, func(t time.Time) tea.Msg {
		return TickMsg{}
	})
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case TickMsg:
		m.ticker++
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
			return m, nil
		case "up", "k":
			if m.cursor > 0 {
				m.cursor--
			} else {
				m.cursor = len(environments) - 1
			}
			if m.cursor == 1 && m.VMStatus == "unknown" {
				m.VMStatus = "checking"
				return m, cloud.CheckVMStatus(m.CloudConfig)
			}
		case "down", "j":
			if m.cursor < len(environments)-1 {
				m.cursor++
			} else {
				m.cursor = 0
			}
			if m.cursor == 1 && m.VMStatus == "unknown" {
				m.VMStatus = "checking"
				return m, cloud.CheckVMStatus(m.CloudConfig)
			}
		case "enter", " ":
			if m.cursor == 1 && m.VMStatus != "running" {
				return m, nil
			}
			m.Selected = environments[m.cursor].env
			m.Done = true
			return m, nil
		case "l", "L":
			m.Selected = EnvLocal
			m.Done = true
			return m, nil
		case "g", "G":
			if m.VMStatus == "running" {
				m.Selected = EnvGoogleCloud
				m.Done = true
			}
			return m, nil
		case "s":
			if m.VMStatus == "stopped" || m.VMStatus == "error" {
				m.VMStatus = "starting"
				return m, cloud.StartVM(m.CloudConfig)
			}
		case "c":
			m.VMStatus = "checking"
			return m, cloud.CheckVMStatus(m.CloudConfig)
		}

	case cloud.VMStatusMsg:
		if msg.Error != nil {
			m.VMStatus = "stopped"
			m.VMError = msg.Error
		} else if msg.Running {
			m.VMStatus = "running"
			m.VMHostname = msg.IP
		} else {
			m.VMStatus = "stopped"
		}
		return m, nil

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
	// Help overlay takes precedence
	if m.showHelp {
		overlay := m.helpOverlay.View()
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, overlay)
	}

	var b strings.Builder

	// Calculate responsive width
	contentWidth := m.width - 6
	if contentWidth < styles.MinContentWidth {
		contentWidth = styles.MinContentWidth
	}
	if contentWidth > styles.MaxContentWidth {
		contentWidth = styles.MaxContentWidth
	}

	separatorWidth := contentWidth - 4
	if separatorWidth > 55 {
		separatorWidth = 55
	}

	// Welcome Banner
	b.WriteString(m.renderWelcomeBanner())
	b.WriteString("\n\n")

	// System Info
	b.WriteString(m.renderSystemInfo())
	b.WriteString("\n\n")

	// Title
	b.WriteString(m.renderTitle(separatorWidth))
	b.WriteString("\n\n")

	// Environment Options
	b.WriteString(m.renderOptions())
	b.WriteString("\n")

	// Cloud details panel when cloud is highlighted
	if m.cursor == 1 {
		b.WriteString(m.renderCloudPanel())
		b.WriteString("\n")
	}

	b.WriteString("\n")

	// Footer
	b.WriteString(m.renderFooter())

	return b.String()
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

func (m Model) renderSystemInfo() string {
	infoStyle := lipgloss.NewStyle().Foreground(styles.TextDim)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	hostname, _ := os.Hostname()
	cpuCount := runtime.NumCPU()

	var info []string
	info = append(info, infoStyle.Render("Host: ")+valueStyle.Render(hostname))
	info = append(info, infoStyle.Render("OS: ")+valueStyle.Render(runtime.GOOS+"/"+runtime.GOARCH))
	info = append(info, infoStyle.Render("CPUs: ")+valueStyle.Render(formatInt(cpuCount)))

	separator := lipgloss.NewStyle().Foreground(styles.Secondary).Render("  │  ")
	return strings.Join(info, separator)
}

func formatInt(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}
	return string(rune('0'+n/10)) + string(rune('0'+n%10))
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
	var b strings.Builder
	nameWidth := 20

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

		// Pad name to fixed width for alignment
		paddedName := opt.name + strings.Repeat(" ", nameWidth-len(opt.name))

		line := cursor + marker + " " + nameStyle.Render(paddedName) + descStyle.Render(opt.description)

		// Add VM status for cloud option
		if opt.env == EnvGoogleCloud {
			line += "  " + m.renderVMStatusBadge()
		}

		b.WriteString(line + "\n")
	}

	return b.String()
}

func (m Model) renderVMStatusBadge() string {
	switch m.VMStatus {
	case "checking", "starting":
		frames := []string{"◐", "◓", "◑", "◒"}
		frame := frames[m.ticker%len(frames)]
		label := "checking"
		if m.VMStatus == "starting" {
			label = "starting"
		}
		return lipgloss.NewStyle().Foreground(styles.Accent).Render("[" + frame + " " + label + "]")
	case "running":
		return lipgloss.NewStyle().Foreground(styles.Success).Render("[" + styles.ActiveMark + " running]")
	case "stopped":
		return lipgloss.NewStyle().Foreground(styles.Warning).Render("[" + styles.PendingMark + " stopped]")
	case "error":
		return lipgloss.NewStyle().Foreground(styles.Error).Render("[" + styles.CrossMark + " error]")
	default:
		return lipgloss.NewStyle().Foreground(styles.Muted).Render("[? unknown]")
	}
}

func (m Model) renderCloudPanel() string {
	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(styles.Secondary).
		Padding(0, 1).
		Width(m.width - 10)

	var content strings.Builder

	labelStyle := lipgloss.NewStyle().Foreground(styles.TextDim).Width(15)
	valueStyle := lipgloss.NewStyle().Foreground(styles.Text)

	content.WriteString(labelStyle.Render("SSH Host:") + valueStyle.Render(m.CloudConfig.RemoteHost) + "\n")
	content.WriteString(labelStyle.Render("Remote Path:") + valueStyle.Render(m.CloudConfig.RemoteBase) + "\n")
	content.WriteString(labelStyle.Render("GCP Instance:") + valueStyle.Render(m.CloudConfig.GCPInstance) + "\n")

	// Status line
	statusLabel := labelStyle.Render("Status:")
	switch m.VMStatus {
	case "running":
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Success).Bold(true).Render(styles.ActiveMark+" Connected"))
		if m.VMHostname != "" {
			content.WriteString(" (" + valueStyle.Render(m.VMHostname) + ")")
		}
		content.WriteString("\n")
	case "stopped":
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Warning).Render(styles.PendingMark+" Stopped"))
		content.WriteString(lipgloss.NewStyle().Foreground(styles.TextDim).Render(" - press [s] to start") + "\n")
	case "checking":
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Accent).Render("Checking connection...") + "\n")
	case "starting":
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Accent).Render("Starting VM...") + "\n")
	case "error":
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Error).Render(styles.CrossMark+" Connection failed") + "\n")
		if m.VMError != nil {
			errMsg := m.VMError.Error()
			if len(errMsg) > 40 {
				errMsg = errMsg[:40] + "..."
			}
			content.WriteString(labelStyle.Render("") + lipgloss.NewStyle().Foreground(styles.TextDim).Render(errMsg) + "\n")
		}
	default:
		content.WriteString(statusLabel + lipgloss.NewStyle().Foreground(styles.Muted).Render("Press [c] to check") + "\n")
	}

	return boxStyle.Render(content.String())
}

func (m Model) renderFooter() string {
	hints := []string{
		styles.RenderKeyHint("↑↓", "Navigate"),
		styles.RenderKeyHint("Enter", "Select"),
		styles.RenderKeyHint("L", "Local"),
		styles.RenderKeyHint("G", "Cloud"),
		styles.RenderKeyHint("C", "Check VM"),
		styles.RenderKeyHint("S", "Start VM"),
		styles.RenderKeyHint("Q", "Quit"),
	}

	separator := "  "
	return styles.FooterStyle.Width(m.width - 8).Render(strings.Join(hints, separator))
}
