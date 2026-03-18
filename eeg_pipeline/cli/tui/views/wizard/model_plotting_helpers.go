package wizard

import "strings"

func (m *Model) applyPlotFormats(formats []string) {
	selected := make(map[string]bool, len(m.plotFormats))
	for _, format := range m.plotFormats {
		selected[format] = false
	}

	for _, format := range formats {
		normalized := strings.ToLower(strings.TrimSpace(format))
		if _, ok := selected[normalized]; ok {
			selected[normalized] = true
		}
	}

	for format, enabled := range selected {
		m.plotFormatSelected[format] = enabled
	}
}

func (m *Model) setPlotDpiIndexFromValue(value int) {
	for i, option := range m.plotDpiOptions {
		if option == value {
			m.plotDpiIndex = i
			return
		}
	}
}

func (m *Model) setPlotSavefigDpiIndexFromValue(value int) {
	for i, option := range m.plotDpiOptions {
		if option == value {
			m.plotSavefigDpiIndex = i
			return
		}
	}
}
