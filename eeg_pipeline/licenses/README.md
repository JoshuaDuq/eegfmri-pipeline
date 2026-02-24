# FreeSurfer License Handling

**Do not commit real FreeSurfer license files to this repository.**

Provide the license path via one of these methods (checked in order):

1. Environment variable: `export EEG_PIPELINE_FREESURFER_LICENSE="$HOME/.freesurfer/license.txt"`
2. Config key: `paths.freesurfer_license` in your local config/overrides.
3. CLI flag: `--fs-license-file /path/to/license.txt`

The file `license_freesurfer.txt.example` is a placeholder template only.
