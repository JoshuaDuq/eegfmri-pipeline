package executor

import "strings"

func JoinCommand(goos string, args []string) string {
	if len(args) == 0 {
		return ""
	}

	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, quoteCommandArg(goos, arg))
	}
	return strings.Join(quoted, " ")
}

func quoteCommandArg(goos string, arg string) string {
	if goos == "windows" {
		return quotePowerShellArg(arg)
	}
	return quotePOSIXArg(arg)
}

func quotePOSIXArg(arg string) string {
	if arg == "" {
		return "''"
	}
	if isPOSIXSafe(arg) {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", `'"'"'`) + "'"
}

func quotePowerShellArg(arg string) string {
	if arg == "" {
		return "''"
	}
	if isPowerShellSafe(arg) {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", "''") + "'"
}

func isPOSIXSafe(arg string) bool {
	for _, r := range arg {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			strings.ContainsRune("@%_+=:,./-~", r) {
			continue
		}
		return false
	}
	return true
}

func isPowerShellSafe(arg string) bool {
	for _, r := range arg {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			strings.ContainsRune("@%_+=:,./-~\\", r) {
			continue
		}
		return false
	}
	return true
}
