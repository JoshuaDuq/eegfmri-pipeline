package styles

///////////////////////////////////////////////////////////////////
// Layout Constants
///////////////////////////////////////////////////////////////////

const (
	MinTerminalWidth   = 60
	MaxTerminalWidth   = 300
	DefaultCardPadding = 2
	MaxVisibleSubjects = 10

	MenuLeftColumnPercent  = 55
	MenuRightColumnPercent = 45
	MinLeftColumnWidth     = 30
	MinRightColumnWidth    = 25

	MinContentWidth     = 50
	MaxContentWidth     = 280
	DefaultContentWidth = 60

	SummaryLabelWidth = 16
	SummaryValueWidth = 30
)

///////////////////////////////////////////////////////////////////
// Execution & Logging Constants
///////////////////////////////////////////////////////////////////

const (
	MaxScrollbackLines = 0 // 0 for unlimited
	LogBufferChannels  = 1000

	ScrollStepSize        = 3
	MouseWheelScrollLines = 1

	DefaultLogHeight = 12
	MinLogHeight     = 8
	MaxLogHeight     = 150
	MinLogWidth      = 60
	MaxLogWidth      = 300

	DefaultProgressBarWidth = 50
	MinProgressBarWidth     = 20
	MaxProgressBarWidth     = 50
	MiniProgressBarWidth    = 35

	TickIntervalMs         = 100
	ProgressStaleWarningMs = 30000
)

///////////////////////////////////////////////////////////////////
// Command & Validation Constants
///////////////////////////////////////////////////////////////////

const (
	DefaultCommandTimeout = 0
	MaxRetryAttempts      = 3
	MinSubjectsRequired   = 1
	MinCategoriesRequired = 1
	MaxRecentErrors       = 5
)

///////////////////////////////////////////////////////////////////
// Status Marks
///////////////////////////////////////////////////////////////////

const (
	CheckMark    = "✓"
	CrossMark    = "✗"
	PendingMark  = "○"
	ActiveMark   = "●"
	SelectedMark = "➜"
	BulletMark   = "•"
	ArrowRight   = "→"
	WarningMark  = "⚠"
)

///////////////////////////////////////////////////////////////////
// Keyboard Shortcuts
///////////////////////////////////////////////////////////////////

const (
	KeyUp     = "↑"
	KeyDown   = "↓"
	KeyLeft   = "←"
	KeyRight  = "→"
	KeyEnter  = "Enter"
	KeyEscape = "Esc"
	KeySpace  = "Space"
	KeyTab    = "Tab"
)

///////////////////////////////////////////////////////////////////
// Mode Constants
///////////////////////////////////////////////////////////////////

const (
	ModeCompute   = "compute"
	ModeVisualize = "visualize"
)

///////////////////////////////////////////////////////////////////
// Divider Styles
///////////////////////////////////////////////////////////////////

type DividerStyle int

const (
	DividerSingle DividerStyle = iota
	DividerDouble
	DividerDashed
	DividerDots
)
