package wizard

import (
	"fmt"
	"time"
)

///////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////

// formatRelativeTime converts ISO timestamp to relative time string
func formatRelativeTime(isoTime string) string {
	if isoTime == "" {
		return ""
	}
	// Try RFC3339 first (standard for ISO with timezone/Z)
	t, err := time.Parse(time.RFC3339, isoTime)
	if err != nil {
		// Try RFC3339Nano (standard for ISO with fractional seconds and timezone/Z)
		t, err = time.Parse(time.RFC3339Nano, isoTime)
		if err != nil {
			// Try without timezone (assume UTC)
			if len(isoTime) >= 19 {
				t, err = time.Parse("2006-01-02T15:04:05", isoTime[:19])
				if err != nil {
					return ""
				}
			} else {
				return ""
			}
		}
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		mins := int(d.Minutes())
		if mins == 1 {
			return "1 min ago"
		}
		return fmt.Sprintf("%d mins ago", mins)
	case d < 24*time.Hour:
		hours := int(d.Hours())
		if hours == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", hours)
	default:
		days := int(d.Hours() / 24)
		if days == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", days)
	}
}
