# TUI Workflow

Use the TUI when you want a guided interface over the same public CLI commands.

## Main Flow

1. Build and launch the Go application.
2. Choose a pipeline from the main menu.
3. Step through the wizard to select subjects, modes, and advanced options.
4. Run the command and watch execution progress live.

## Build And Run

```bash
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

## What The TUI Adds

- subject discovery and status badges
- guided pipeline configuration
- live execution monitoring
- persistent project overrides and run history

## Reference

Architecture, views, keyboard shortcuts, and persistence details:
[TUI reference](../reference/tui.md)
