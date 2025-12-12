"""
Allow running the package as: python -m eeg_pipeline
"""

from eeg_pipeline.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
