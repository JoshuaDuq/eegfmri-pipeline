package styles

const (
	MinTerminalWidth  = 60
	MinTerminalHeight = 20
	MaxTerminalWidth  = 300

	MinContentWidth = 50
	MaxContentWidth = 280

	MinListItems     = 5
	ListScrollMargin = 2
)

const (
	MaxScrollbackLines = 0
	LogBufferChannels  = 1000

	ScrollStepSize        = 3
	MouseWheelScrollLines = 1

	DefaultLogHeight = 12
	MinLogHeight     = 8
	MaxLogHeight     = 150
	MinLogWidth      = 60
	MaxLogWidth      = 300

	MinProgressBarWidth = 20
	MaxProgressBarWidth = 50

	TickIntervalMs = 100

	ResourceMonitorIntervalSec = 1

	ExecHeaderLines            = 2
	ExecInfoPanelLines         = 6
	ExecProgressSectionLines   = 8
	ExecLogTitleLines          = 2
	ExecViewportBorderLines    = 2
	ExecFooterLines            = 4
	ExecBaseReservedLines      = ExecHeaderLines + ExecInfoPanelLines + ExecProgressSectionLines + ExecLogTitleLines + ExecViewportBorderLines + ExecFooterLines
	ExecMetricsDashboardLines  = 8
	ExecCompletionSummaryLines = 14
	ExecCopyModeBannerLines    = 1
	ExecSearchInputLines       = 1

	MaxRecentErrors = 5
)

const (
	CheckMark           = "✓"
	CrossMark           = "✗"
	PendingMark         = "○"
	ActiveMark          = "●"
	SelectedMark        = "▸"
	BulletMark          = "•"
	WarningMark         = "⚠"
	FooterHintSeparator = "  │  "

	HeaderSeparatorChar = "─"
	SectionDividerChar  = "─"
)

const (
	ModeCompute   = "compute"
	ModeVisualize = "visualize"
)

type DividerStyle int

const (
	DividerSingle DividerStyle = iota
	DividerDouble
	DividerDashed
	DividerDots
)
