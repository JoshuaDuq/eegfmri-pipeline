package styles

import (
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

var stylesANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func stripStylesANSI(s string) string {
	return stylesANSIPattern.ReplaceAllString(s, "")
}

func TestListLayoutHelpers(t *testing.T) {
	if got := calculateAvailableRows(4, 1); got != MinListItems {
		t.Fatalf("calculateAvailableRows() = %d, want %d", got, MinListItems)
	}

	layout := CalculateListLayout(12, 8, 20, 2)
	if layout.MaxItems != 10 {
		t.Fatalf("MaxItems = %d, want 10", layout.MaxItems)
	}
	if layout.StartIdx != 3 || layout.EndIdx != 13 {
		t.Fatalf("unexpected scroll window: %+v", layout)
	}
	if !layout.ShowScrollUp || !layout.ShowScrollDn {
		t.Fatalf("expected both scroll indicators to be visible: %+v", layout)
	}

	if got := calculateStartIndex(1, 20, 10); got != 0 {
		t.Fatalf("calculateStartIndex() = %d, want 0", got)
	}
	if got := calculateStartIndex(19, 20, 10); got != 10 {
		t.Fatalf("calculateStartIndex() = %d, want 10", got)
	}
	if got := clampStartIndex(-3, 20, 10); got != 0 {
		t.Fatalf("clampStartIndex() = %d, want 0", got)
	}
	if got := clampEndIndex(15, 10, 20); got != 20 {
		t.Fatalf("clampEndIndex() = %d, want 20", got)
	}
}

func TestIndicatorAndLayoutRenderers(t *testing.T) {
	if got := stripStylesANSI(RenderScrollUpIndicator(0)); got != "" {
		t.Fatalf("RenderScrollUpIndicator(0) = %q", got)
	}
	if got := stripStylesANSI(RenderScrollUpIndicator(2)); got != "  ▲ 2 more" {
		t.Fatalf("RenderScrollUpIndicator(2) = %q", got)
	}
	if got := stripStylesANSI(RenderScrollDownIndicator(3)); got != "  ▼ 3 more" {
		t.Fatalf("RenderScrollDownIndicator(3) = %q", got)
	}

	if !IsTerminalTooSmall(MinTerminalWidth-1, MinTerminalHeight) {
		t.Fatal("expected terminal to be too small")
	}
	if IsTerminalTooSmall(MinTerminalWidth, MinTerminalHeight) {
		t.Fatal("expected minimum terminal size to be accepted")
	}

	tooSmall := stripStylesANSI(RenderTerminalTooSmall(50, 10))
	if !strings.Contains(tooSmall, "Terminal too small") || !strings.Contains(tooSmall, "60x20") {
		t.Fatalf("unexpected terminal-too-small message: %q", tooSmall)
	}

	if got := RenderCursorOptional(false); got != "  " {
		t.Fatalf("RenderCursorOptional(false) = %q", got)
	}
	cursor := stripStylesANSI(RenderCursor())
	if !strings.Contains(cursor, SelectedMark) {
		t.Fatalf("RenderCursor() = %q", cursor)
	}
}

func TestFormattingRenderers(t *testing.T) {
	if got := stripStylesANSI(RenderHeaderSeparator(4)); got != "────" {
		t.Fatalf("RenderHeaderSeparator(4) = %q", got)
	}
	if got := RenderHeaderSeparator(0); got != "" {
		t.Fatalf("RenderHeaderSeparator(0) = %q", got)
	}

	if got := stripStylesANSI(RenderCheckbox(true, true)); got != "▣" {
		t.Fatalf("RenderCheckbox(true, true) = %q", got)
	}
	if got := stripStylesANSI(RenderCheckbox(false, false)); got != "□" {
		t.Fatalf("RenderCheckbox(false, false) = %q", got)
	}
	if got := stripStylesANSI(RenderRadio(true, false)); got != "◉" {
		t.Fatalf("RenderRadio(true, false) = %q", got)
	}

	if got := stripStylesANSI(RenderSectionLabel("Alpha")); !strings.Contains(got, "Alpha") {
		t.Fatalf("RenderSectionLabel() = %q", got)
	}
	if got := stripStylesANSI(RenderDimSectionLabel("Beta")); !strings.Contains(got, "Beta") {
		t.Fatalf("RenderDimSectionLabel() = %q", got)
	}
	if got := stripStylesANSI(RenderPreviewSubHeader("Gamma")); !strings.Contains(got, "GAMMA") {
		t.Fatalf("RenderPreviewSubHeader() = %q", got)
	}
	if got := stripStylesANSI(RenderPreviewSubHeaderWithRule("Gamma", 12)); !strings.Contains(got, "GAMMA") || !strings.Contains(got, "─") {
		t.Fatalf("RenderPreviewSubHeaderWithRule() = %q", got)
	}
	if got := stripStylesANSI(RenderSectionBlock("Delta", 3)); !strings.Contains(got, "Delta") || !strings.Contains(got, "───") {
		t.Fatalf("RenderSectionBlock() = %q", got)
	}

	if got := stripStylesANSI(RenderKeyValue("Key", "Value", 8)); !strings.Contains(got, "Key") || !strings.Contains(got, "Value") {
		t.Fatalf("RenderKeyValue() = %q", got)
	}
	if got := stripStylesANSI(RenderKeyValueAccent("Key", "Value", 8)); !strings.Contains(got, "Value") {
		t.Fatalf("RenderKeyValueAccent() = %q", got)
	}

	if got := RenderDivider(0); got != "" {
		t.Fatalf("RenderDivider(0) = %q", got)
	}
	if got := stripStylesANSI(RenderDivider(3)); got != "───" {
		t.Fatalf("RenderDivider(3) = %q", got)
	}
	if got := stripStylesANSI(RenderFooterDivider(3)); got != "───" {
		t.Fatalf("RenderFooterDivider(3) = %q", got)
	}
	if got := RenderFooterDivider(0); got != "" {
		t.Fatalf("RenderFooterDivider(0) = %q", got)
	}

	if got := TruncateLine("abcdef", 3); got != "abc" {
		t.Fatalf("TruncateLine(3) = %q", got)
	}
	if got := TruncateLine("abcdef", 4); got != "a..." {
		t.Fatalf("TruncateLine(4) = %q", got)
	}
	if got := PadRight("abc", 5); got != "abc  " {
		t.Fatalf("PadRight() = %q", got)
	}
	if got := FitLine("abcdef", 4); got != "a..." {
		t.Fatalf("FitLine() = %q", got)
	}
	if got := FitLine("abc", 5); got != "abc  " {
		t.Fatalf("FitLine() = %q", got)
	}

	line := stripStylesANSI(RenderConfigLine("▸ ", "Label:", "Value", "Hint", 8, 80))
	if !strings.Contains(line, "Label:") || !strings.Contains(line, "Value") || !strings.Contains(line, "Hint") {
		t.Fatalf("RenderConfigLine() = %q", line)
	}

	if got := stripStylesANSI(RenderStepHeader("Step", 4)); !strings.Contains(got, "Step") || !strings.Contains(got, "────") {
		t.Fatalf("RenderStepHeader() = %q", got)
	}

	if got := stripStylesANSI(RenderStatusCount(0, 3, "subjects")); !strings.Contains(got, "0/3") || !strings.Contains(got, "select at least 1") {
		t.Fatalf("RenderStatusCount(0, 3) = %q", got)
	}
	if got := stripStylesANSI(RenderStatusCount(2, 3, "subjects")); !strings.Contains(got, "2/3") {
		t.Fatalf("RenderStatusCount(2, 3) = %q", got)
	}
}

func TestRenderFooterHintsUsesDotSeparator(t *testing.T) {
	hints := []FooterHint{
		{Key: "A", Label: "Alpha", Priority: 0},
		{Key: "B", Label: "Beta", Priority: 0},
	}

	rendered := stripStylesANSI(RenderFooterHints(80, hints))
	if !strings.Contains(rendered, "·") {
		t.Fatalf("expected footer hints to use a dot separator, got %q", rendered)
	}
	if strings.Contains(rendered, "│") {
		t.Fatalf("expected footer hints to avoid the heavy separator bar, got %q", rendered)
	}
}

func TestClampBlockAndRenderNoWrapBlock(t *testing.T) {
	block := ClampBlock("123456\nabc", 4)
	lines := strings.Split(block, "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 lines, got %d: %q", len(lines), block)
	}
	if lines[0] != "1..." {
		t.Fatalf("expected first line to truncate, got %q", lines[0])
	}
	if lines[1] != "abc" {
		t.Fatalf("expected second line unchanged, got %q", lines[1])
	}

	rendered := stripStylesANSI(RenderNoWrapBlock(lipgloss.NewStyle().Padding(0, 1), "123456", 6))
	renderedLines := strings.Split(rendered, "\n")
	if len(renderedLines) != 1 {
		t.Fatalf("expected single rendered line, got %d: %q", len(renderedLines), rendered)
	}
	if got := lipgloss.Width(renderedLines[0]); got != 6 {
		t.Fatalf("expected rendered width 6, got %d: %q", got, renderedLines[0])
	}
}

func TestRenderProgressBar(t *testing.T) {
	// Width clamp: width below MinProgressBarWidth is raised to MinProgressBarWidth
	got := stripStylesANSI(RenderProgressBar(0.5, 5))
	if !strings.Contains(got, "50%") {
		t.Fatalf("RenderProgressBar(0.5, 5) = %q", got)
	}
	if strings.Count(got, "█") != MinProgressBarWidth/2 {
		t.Fatalf("RenderProgressBar(0.5, 5): expected %d filled cells, got %q", MinProgressBarWidth/2, got)
	}

	// Zero progress: no filled cells
	zero := stripStylesANSI(RenderProgressBar(0, MinProgressBarWidth))
	if strings.Contains(zero, "█") {
		t.Fatalf("RenderProgressBar(0) should have no filled cells, got %q", zero)
	}
	if !strings.Contains(zero, "0%") {
		t.Fatalf("RenderProgressBar(0) should contain 0%%, got %q", zero)
	}

	// Full progress: no empty cells
	full := stripStylesANSI(RenderProgressBar(1.0, MinProgressBarWidth))
	if strings.Contains(full, "░") {
		t.Fatalf("RenderProgressBar(1.0) should have no empty cells, got %q", full)
	}
	if !strings.Contains(full, "100%") {
		t.Fatalf("RenderProgressBar(1.0) should contain 100%%, got %q", full)
	}

	// Clamping: values outside [0, 1] behave like boundary values
	if stripStylesANSI(RenderProgressBar(-0.5, MinProgressBarWidth)) != stripStylesANSI(RenderProgressBar(0, MinProgressBarWidth)) {
		t.Fatal("RenderProgressBar(-0.5) should equal RenderProgressBar(0)")
	}
	if stripStylesANSI(RenderProgressBar(1.5, MinProgressBarWidth)) != stripStylesANSI(RenderProgressBar(1.0, MinProgressBarWidth)) {
		t.Fatal("RenderProgressBar(1.5) should equal RenderProgressBar(1.0)")
	}
}

func TestRenderStepAndPreviewHeadersUseQuieterChrome(t *testing.T) {
	stepHeader := stripStylesANSI(RenderStepHeader("Step", 4))
	if strings.Contains(stepHeader, "\n") {
		t.Fatalf("expected RenderStepHeader() to stay on one line, got %q", stepHeader)
	}
	if !strings.Contains(stepHeader, "Step") {
		t.Fatalf("expected RenderStepHeader() to include the title, got %q", stepHeader)
	}

	previewHeader := stripStylesANSI(RenderPreviewSubHeader("Gamma"))
	if strings.Count(previewHeader, "─") != 4 {
		t.Fatalf("expected RenderPreviewSubHeader() to use a shorter 4-char rule, got %q", previewHeader)
	}
	if !strings.Contains(previewHeader, "·") {
		t.Fatalf("expected RenderPreviewSubHeader() to include a middle-dot lead-in, got %q", previewHeader)
	}
	if !strings.Contains(previewHeader, "GAMMA") {
		t.Fatalf("expected RenderPreviewSubHeader() to uppercase the title, got %q", previewHeader)
	}
}

func TestRenderSectionLabelUsesThinMarker(t *testing.T) {
	label := stripStylesANSI(RenderSectionLabel("Alpha"))
	if strings.Contains(label, "┃") || strings.Contains(label, "│") {
		t.Fatalf("expected RenderSectionLabel() to use the hairline section marker, got %q", label)
	}
	if !strings.Contains(label, "▏") {
		t.Fatalf("expected RenderSectionLabel() to use the ▏ hairline marker, got %q", label)
	}
}
