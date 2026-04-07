//go:build !windows

package execution

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/eeg-pipeline/tui/messages"
)

func sampleResourceUpdate(
	pid int,
	numCores int,
	prevCoreTimes []CPUCoreTimes,
) (messages.ResourceUpdateMsg, []CPUCoreTimes) {
	cpu, mem := getProcessResources(pid)
	coreUsages, nextTimes := calculatePerCoreCPUUsage(numCores, prevCoreTimes)
	return messages.ResourceUpdateMsg{
		CPUUsage:      cpu,
		CPUAvailable:  true,
		MemoryUsage:   mem,
		CPUCoreUsages: coreUsages,
		NumCPUCores:   numCores,
	}, nextTimes
}

func getProcessResources(pid int) (float64, float64) {
	cmd := exec.Command("ps", "-p", fmt.Sprintf("%d", pid), "-o", "%cpu=,rss=")
	output, err := cmd.Output()
	if err != nil {
		return 0.0, 0.0
	}

	fields := strings.Fields(strings.TrimSpace(string(output)))
	if len(fields) < 2 {
		return 0.0, 0.0
	}

	var cpuUsage float64
	var memKB float64
	fmt.Sscanf(fields[0], "%f", &cpuUsage)
	fmt.Sscanf(fields[1], "%f", &memKB)

	memGB := memKB / (1024 * 1024)
	return cpuUsage, memGB
}

func getSystemCPUTimes(numCores int) []CPUCoreTimes {
	times := make([]CPUCoreTimes, numCores)

	data, err := os.ReadFile("/proc/stat")
	if err == nil {
		lines := strings.Split(string(data), "\n")
		coreIdx := 0
		for _, line := range lines {
			if strings.HasPrefix(line, "cpu") && !strings.HasPrefix(line, "cpu ") {
				fields := strings.Fields(line)
				if len(fields) >= 5 && coreIdx < numCores {
					fmt.Sscanf(fields[1], "%d", &times[coreIdx].User)
					fmt.Sscanf(fields[2], "%d", &times[coreIdx].Nice)
					fmt.Sscanf(fields[3], "%d", &times[coreIdx].System)
					fmt.Sscanf(fields[4], "%d", &times[coreIdx].Idle)
					coreIdx++
				}
			}
		}
		return times
	}

	cmd := exec.Command("top", "-l", "1", "-n", "0", "-stats", "cpu")
	output, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "CPU usage:") {
				var user, sys, idle float64
				line = strings.TrimPrefix(line, "CPU usage:")
				parts := strings.Split(line, ",")
				for _, part := range parts {
					part = strings.TrimSpace(part)
					if strings.Contains(part, "user") {
						fmt.Sscanf(part, "%f%% user", &user)
					} else if strings.Contains(part, "sys") {
						fmt.Sscanf(part, "%f%% sys", &sys)
					} else if strings.Contains(part, "idle") {
						fmt.Sscanf(part, "%f%% idle", &idle)
					}
				}
				for i := 0; i < numCores; i++ {
					times[i].User = uint64(user * 100)
					times[i].System = uint64(sys * 100)
					times[i].Idle = uint64(idle * 100)
				}
				break
			}
		}
	}

	return times
}

func calculatePerCoreCPUUsage(numCores int, prevTimes []CPUCoreTimes) ([]float64, []CPUCoreTimes) {
	currentTimes := getSystemCPUTimes(numCores)
	usages := make([]float64, numCores)

	if _, err := os.Stat("/proc/stat"); err == nil {
		for i := 0; i < numCores; i++ {
			if i < len(prevTimes) && i < len(currentTimes) {
				prevTotal := prevTimes[i].User + prevTimes[i].System + prevTimes[i].Idle + prevTimes[i].Nice
				currTotal := currentTimes[i].User + currentTimes[i].System + currentTimes[i].Idle + currentTimes[i].Nice
				prevIdle := prevTimes[i].Idle
				currIdle := currentTimes[i].Idle

				totalDelta := currTotal - prevTotal
				idleDelta := currIdle - prevIdle

				if totalDelta > 0 {
					usages[i] = float64(totalDelta-idleDelta) / float64(totalDelta) * 100.0
				}
			}
		}
		return usages, currentTimes
	}

	return getPerCoreUsageMacOS(numCores), currentTimes
}

func getPerCoreUsageMacOS(numCores int) []float64 {
	usages := make([]float64, numCores)

	cmd := exec.Command("top", "-l", "1", "-n", "0")
	output, err := cmd.Output()
	if err != nil {
		return usages
	}

	var userPct, sysPct float64
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "CPU usage:") {
			parts := strings.Split(strings.TrimPrefix(line, "CPU usage:"), ",")
			for _, part := range parts {
				part = strings.TrimSpace(part)
				if strings.Contains(part, "user") {
					fmt.Sscanf(part, "%f%% user", &userPct)
				} else if strings.Contains(part, "sys") {
					fmt.Sscanf(part, "%f%% sys", &sysPct)
				}
			}
			break
		}
	}

	totalUsage := userPct + sysPct
	if totalUsage <= 0 {
		return usages
	}

	psCmd := exec.Command("ps", "-A", "-o", "%cpu=")
	psOutput, err := psCmd.Output()
	if err != nil {
		for i := 0; i < numCores; i++ {
			usages[i] = totalUsage / float64(numCores)
		}
		return usages
	}

	var processTotal float64
	for _, line := range strings.Split(string(psOutput), "\n") {
		value := strings.TrimSpace(line)
		if value == "" {
			continue
		}
		var cpu float64
		if _, scanErr := fmt.Sscanf(value, "%f", &cpu); scanErr == nil {
			processTotal += cpu
		}
	}
	if processTotal <= 0 {
		for i := 0; i < numCores; i++ {
			usages[i] = totalUsage / float64(numCores)
		}
		return usages
	}

	for i := 0; i < numCores; i++ {
		usages[i] = totalUsage / float64(numCores)
	}
	return usages
}
