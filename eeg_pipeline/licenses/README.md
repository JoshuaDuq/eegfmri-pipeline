# FreeSurfer License Handling

**Do not commit real FreeSurfer license files to this repository.**

Recommended default location: `~/license.txt`.

License path resolution order:

1. CLI flag: `--fs-license-file /path/to/license.txt`
2. Config key: `paths.freesurfer_license` in your local config/overrides
3. Environment variable: `export EEG_PIPELINE_FREESURFER_LICENSE="$HOME/license.txt"`
4. Default file path: `~/license.txt` (used automatically if none of the above are set)

The file `license_freesurfer.txt.example` is a placeholder template only.
