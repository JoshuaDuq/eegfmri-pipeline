package execution

import (
	"runtime"
	"time"

	"github.com/eeg-pipeline/tui/messages"
	"github.com/eeg-pipeline/tui/styles"

	tea "github.com/charmbracelet/bubbletea"
)

type CPUCoreTimes struct {
	User   uint64
	System uint64
	Idle   uint64
	Nice   uint64
}

// Runtime CPU/memory sampling and per-core usage helpers.

func (m *Model) startResourceMonitoring() tea.Cmd {
	return func() tea.Msg {
		if m.cmd == nil || m.cmd.Process == nil || m.resourceUpdateChan == nil || m.stopResourceChan == nil {
			return nil
		}

		pid := m.cmd.Process.Pid
		updateChan := m.resourceUpdateChan
		stopChan := m.stopResourceChan
		numCores := runtime.NumCPU()
		ticker := time.NewTicker(time.Duration(styles.ResourceMonitorIntervalSec) * time.Second)
		defer ticker.Stop()

		safeSend := func(msg messages.ResourceUpdateMsg) bool {
			defer func() {
				recover()
			}()
			select {
			case <-stopChan:
				return false
			case updateChan <- msg:
				return true
			default:
				return true
			}
		}

		prevCoreTimes := getSystemCPUTimes(numCores)

		initialUpdate, nextCoreTimes := sampleResourceUpdate(pid, numCores, prevCoreTimes)
		prevCoreTimes = nextCoreTimes
		if !safeSend(initialUpdate) {
			return nil
		}

		for {
			select {
			case <-stopChan:
				return nil
			case <-ticker.C:
				if !processIsRunning(m.cmd) {
					return nil
				}
				update, nextTimes := sampleResourceUpdate(pid, numCores, prevCoreTimes)
				prevCoreTimes = nextTimes
				if !safeSend(update) {
					return nil
				}
			}
		}
	}
}
