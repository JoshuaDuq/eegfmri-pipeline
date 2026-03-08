package styles

const (
	NarrowLayoutWidth = 100
	WideLayoutWidth   = 120
	TallLayoutHeight  = 28
)

func IsNarrowLayout(width int) bool {
	return width < NarrowLayoutWidth
}

func UseTwoColumnLayout(width, height int) bool {
	return width >= WideLayoutWidth && height >= TallLayoutHeight
}
