package wizard

import (
	"fmt"
	"time"
)

// formatRelativeTime converts ISO timestamp to relative time string.
// Returns empty string if timestamp cannot be parsed.
func formatRelativeTime(isoTime string) string {
	if isoTime == "" {
		return ""
	}

	parsedTime := parseISOTimestamp(isoTime)
	if parsedTime.IsZero() {
		return ""
	}

	return formatTimeAgo(time.Since(parsedTime))
}

// parseISOTimestamp parses ISO timestamp in RFC3339 formats.
// Returns zero time if parsing fails.
func parseISOTimestamp(isoTime string) time.Time {
	formats := []string{
		time.RFC3339Nano,
		time.RFC3339,
	}

	for _, format := range formats {
		if parsedTime, err := time.Parse(format, isoTime); err == nil {
			return parsedTime
		}
	}

	return time.Time{}
}

// formatTimeAgo formats duration as human-readable relative time string.
func formatTimeAgo(duration time.Duration) string {
	switch {
	case duration < time.Minute:
		return "just now"
	case duration < time.Hour:
		return formatMinutes(duration)
	case duration < 24*time.Hour:
		return formatHours(duration)
	default:
		return formatDays(duration)
	}
}

// formatMinutes formats duration as minutes ago string.
func formatMinutes(duration time.Duration) string {
	minutes := int(duration.Minutes())
	if minutes == 1 {
		return "1 min ago"
	}
	return fmt.Sprintf("%d mins ago", minutes)
}

// formatHours formats duration as hours ago string.
func formatHours(duration time.Duration) string {
	hours := int(duration.Hours())
	if hours == 1 {
		return "1 hour ago"
	}
	return fmt.Sprintf("%d hours ago", hours)
}

// formatDays formats duration as days ago string.
func formatDays(duration time.Duration) string {
	days := int(duration.Hours() / 24)
	if days == 1 {
		return "1 day ago"
	}
	return fmt.Sprintf("%d days ago", days)
}
