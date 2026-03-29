# Getting Started

This page is the shortest path from a fresh clone to a working install.
For the complete, copy-paste workflow and the full command matrix, use
[Quick Start](../user_guide/quickstart).

## Installation

Requirements:

- Python 3.11 or newer
- A virtual environment
- Go 1.21+ only if you want the optional TUI

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ml]"
```

## Smoke Test

Validate inputs and confirm subject discovery:

```bash
eeg-pipeline validate quick
eeg-pipeline info subjects
eeg-pipeline info config
eeg-pipeline info ml-feature-space
```

## Next

- Canonical walkthrough: [Quick Start](../user_guide/quickstart)
- Command matrix and flags: [CLI Reference](../user_guide/cli/index)
- Optional guided interface: [TUI](../user_guide/tui)
- Install and environment details: [Install](../install)

## Optional TUI

The Go TUI wraps the same CLI and executes commands from the repository root.

```bash
cd eeg_pipeline/cli/tui
go build -o eeg-tui .
./eeg-tui
```

## Documentation Build

```bash
python -m pip install -e ".[docs]"
make docs
make docs-check
```
