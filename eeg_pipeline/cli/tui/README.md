# TUI (Text User Interface)

A terminal-based user interface for the EEG Pipeline built with Go and [Bubble Tea](https://github.com/charmbracelet/bubbletea).

## Prerequisites

- Go 1.21 or later
- Python environment with the EEG pipeline dependencies installed

## Building and Running

### From a Fresh Terminal

Navigate to the TUI directory and build/run:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/cli/tui && go mod download && go build -o tui . && ./tui
```

### Step-by-Step

1. **Navigate to TUI directory:**
   ```bash
   cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/cli/tui
   ```

2. **Download dependencies:**
   ```bash
   go mod download
   ```
   (Or `go mod tidy` to also clean up unused dependencies)

3. **Build the binary:**
   ```bash
   go build -o tui .
   ```
   This creates an executable named `tui` in the current directory.

4. **Run it:**
   ```bash
   ./tui
   ```

### Alternative: Run Directly Without Building

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline/eeg_pipeline/cli/tui && go run main.go
```

## Notes

- The TUI expects to be run from the repository root context (it searches for the `eeg_pipeline` directory to find the repo root).
- Ensure Go 1.21+ is installed. Check with `go version`.
