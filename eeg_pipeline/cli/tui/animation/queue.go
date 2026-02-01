package animation

// Kind identifies an animation type for rendering (e.g. cursor blink, progress pulse).
type Kind int

const (
	KindNone Kind = iota
	KindCursorBlink
	KindProgressPulse
)

// Item is a single animation in the queue (kind, duration in ticks, optional loop).
type Item struct {
	Kind          Kind
	DurationTicks int
	Loop          bool
}

// Queue runs a sequence of animations; one item is active at a time and advanced each Tick.
type Queue struct {
	items       []Item
	currentTick int
}

// Push appends an animation to the queue.
func (q *Queue) Push(item Item) {
	q.items = append(q.items, item)
}

// Tick advances the current animation by one tick. When duration is reached, the item
// is reset (if Loop) or removed. Returns true if the queue is still active.
func (q *Queue) Tick() bool {
	if len(q.items) == 0 {
		return false
	}
	q.currentTick++
	cur := &q.items[0]
	if q.currentTick >= cur.DurationTicks {
		if cur.Loop {
			q.currentTick = 0
		} else {
			q.items = q.items[1:]
			q.currentTick = 0
		}
	}
	return true
}

// Current returns the active animation kind and progress in [0, 1].
func (q *Queue) Current() (Kind, float64) {
	if len(q.items) == 0 {
		return KindNone, 0
	}
	cur := q.items[0]
	progress := float64(q.currentTick) / float64(cur.DurationTicks)
	if progress > 1 {
		progress = 1
	}
	return cur.Kind, progress
}

// CursorVisible returns true when the current animation is cursor blink and progress < 0.5
// (show cursor for first half of cycle, hide for second half).
func (q *Queue) CursorVisible() bool {
	kind, progress := q.Current()
	return kind == KindCursorBlink && progress < 0.5
}

// Active returns whether the queue has any items.
func (q *Queue) Active() bool {
	return len(q.items) > 0
}

// CursorBlinkLoop returns an item for a looping cursor blink (e.g. 5 ticks visible, 5 hidden at 100ms/tick).
func CursorBlinkLoop() Item {
	return Item{Kind: KindCursorBlink, DurationTicks: 10, Loop: true}
}

// ProgressPulseLoop returns an item for a looping progress/loading pulse (e.g. leading-edge or icon pulse).
func ProgressPulseLoop() Item {
	return Item{Kind: KindProgressPulse, DurationTicks: 10, Loop: true}
}
