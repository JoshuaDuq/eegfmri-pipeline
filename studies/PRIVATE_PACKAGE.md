# Private Studies Package

`studies/` is now structured to be removed from the public wheel and registered
as a private extension package for `eeg-pipeline`.

When you extract this directory into a private repository, the private package
should expose the study commands through the `eeg_pipeline.cli_commands`
entry-point group.

Minimal package metadata for the private repository:

```toml
[project]
name = "eeg-pipeline-studies"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["eeg-pipeline"]

[project.entry-points."eeg_pipeline.cli_commands"]
coupling = "studies.pain_study.cli.command_registry:coupling_command"
signature-prediction = "studies.pain_study.cli.command_registry:signature_prediction_command"
source-interpretation = "studies.pain_study.cli.command_registry:source_interpretation_command"

[tool.setuptools.packages.find]
include = ["studies*"]

[tool.setuptools.package-data]
"studies.pain_study" = ["scripts/config/*.yaml"]
"studies.pain_study.study1" = ["config/*.yaml", "README.md"]
"studies.pain_study.study2" = ["config/*.yaml", "README.md"]
"studies.pain_study.eeg_coupling" = [
    "analysis/*.R",
    "config/*.yaml",
    "config/**/*.json",
    "config/**/*.label",
    "README.md",
]
```

The public repository should not package `studies/`. The private package owns
the study-specific CLI registration and assets.

Local private-only files added under `studies/`:

- `studies/private_package.toml`: minimal private package metadata scaffold
- `studies/pytest.ini`: private pytest configuration
- `studies/tests/`: recovered study-specific test suite

Run the private suite locally with:

```bash
.venv/bin/python -m pytest --no-cov -c studies/pytest.ini studies/tests -q
```
