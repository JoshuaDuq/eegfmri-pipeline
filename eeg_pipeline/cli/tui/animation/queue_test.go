package animation

import "testing"

func TestQueueLifecycle(t *testing.T) {
	var q Queue
	if q.Active() {
		t.Fatal("expected empty queue to be inactive")
	}
	if kind, progress := q.Current(); kind != KindNone || progress != 0 {
		t.Fatalf("unexpected empty current state: kind=%v progress=%v", kind, progress)
	}
	if q.CursorVisible() {
		t.Fatal("expected empty queue cursor to be hidden")
	}

	q.Push(Item{Kind: KindCursorBlink, DurationTicks: 2, Loop: true})
	if !q.Active() {
		t.Fatal("expected queue to become active after Push")
	}
	if kind, progress := q.Current(); kind != KindCursorBlink || progress != 0 {
		t.Fatalf("unexpected current state after push: kind=%v progress=%v", kind, progress)
	}
	if !q.CursorVisible() {
		t.Fatal("expected cursor blink to be visible at start of cycle")
	}

	if !q.Tick() {
		t.Fatal("expected looping queue to remain active after Tick")
	}
	if kind, progress := q.Current(); kind != KindCursorBlink || progress != 0.5 {
		t.Fatalf("unexpected looping progress after first tick: kind=%v progress=%v", kind, progress)
	}
	if q.CursorVisible() {
		t.Fatal("expected cursor to hide in second half of cycle")
	}

	if !q.Tick() {
		t.Fatal("expected looping queue to remain active after second Tick")
	}
	if kind, progress := q.Current(); kind != KindCursorBlink || progress != 0 {
		t.Fatalf("expected looping queue to reset progress, got kind=%v progress=%v", kind, progress)
	}

	q2 := Queue{}
	q2.Push(Item{Kind: KindProgressPulse, DurationTicks: 2})
	if !q2.Tick() {
		t.Fatal("expected non-looping queue to remain active on first tick")
	}
	if kind, progress := q2.Current(); kind != KindProgressPulse || progress != 0.5 {
		t.Fatalf("unexpected non-looping progress after first tick: kind=%v progress=%v", kind, progress)
	}
	if !q2.Tick() {
		t.Fatal("expected non-looping queue to remain active until duration elapses")
	}
	if q2.Active() {
		t.Fatal("expected non-looping queue to be removed after duration")
	}
	if q2.Tick() {
		t.Fatal("expected empty queue Tick to return false")
	}
}

func TestLoopFactories(t *testing.T) {
	if got := CursorBlinkLoop(); got.Kind != KindCursorBlink || got.DurationTicks != 10 || !got.Loop {
		t.Fatalf("unexpected cursor blink loop: %+v", got)
	}
	if got := ProgressPulseLoop(); got.Kind != KindProgressPulse || got.DurationTicks != 10 || !got.Loop {
		t.Fatalf("unexpected progress pulse loop: %+v", got)
	}
}

