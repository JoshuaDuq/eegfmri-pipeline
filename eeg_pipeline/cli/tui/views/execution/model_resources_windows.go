//go:build windows

package execution

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/eeg-pipeline/tui/messages"
)

func sampleResourceUpdate(
	pid int,
	numCores int,
	prevCoreTimes []CPUCoreTimes,
) (messages.ResourceUpdateMsg, []CPUCoreTimes) {
	return messages.ResourceUpdateMsg{
		CPUUsage:      0,
		CPUAvailable:  false,
		MemoryUsage:   getProcessMemoryGB(pid),
		CPUCoreUsages: nil,
		NumCPUCores:   0,
	}, prevCoreTimes
}

func getSystemCPUTimes(numCores int) []CPUCoreTimes {
	return make([]CPUCoreTimes, numCores)
}

func getProcessMemoryGB(pid int) float64 {
	command := fmt.Sprintf(
		"$p = Get-Process -Id %d -ErrorAction SilentlyContinue; if ($null -eq $p) { exit 1 }; [Console]::Write($p.WorkingSet64)",
		pid,
	)
	cmd := exec.Command("powershell", "-NoProfile", "-Command", command)
	output, err := cmd.Output()
	if err != nil {
		return 0
	}

	bytesValue, err := strconv.ParseFloat(strings.TrimSpace(string(output)), 64)
	if err != nil {
		return 0
	}
	return bytesValue / (1024 * 1024 * 1024)
}
