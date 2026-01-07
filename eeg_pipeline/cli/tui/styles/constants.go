package styles

///////////////////////////////////////////////////////////////////
// Layout Constants
///////////////////////////////////////////////////////////////////

const (
	MinTerminalWidth   = 60
	MinTerminalHeight  = 20
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

	// Responsive layout constants
	NarrowThreshold    = 100 // Width below which to use narrow layout
	ShortThreshold     = 25  // Height below which to use compact layout
	VeryShortThreshold = 20  // Height below which to hide optional elements

	// List display constants
	MinListItems     = 5 // Minimum items to show in a scrollable list
	DefaultListItems = 10
	ListScrollMargin = 2 // Lines kept visible above/below cursor when scrolling
	HeaderFooterRows = 6 // Base reserved rows (header ~3 + footer ~3), views add more as needed
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

	// Resource monitoring interval (in seconds)
	// Lower values = more frequent updates but slightly more overhead
	// 1-2 seconds is a good balance for live updates
	ResourceMonitorIntervalSec = 1
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
