## FreeSurfer License Handling

Do not commit real FreeSurfer license files to this repository.

Preferred setup:

1. Export `EEG_PIPELINE_FREESURFER_LICENSE` to an absolute path on your machine.
2. Or set `paths.freesurfer_license` in your local config/overrides.
3. Or pass `--fs-license-file` on CLI calls that need it.

Example:

```bash
export EEG_PIPELINE_FREESURFER_LICENSE="$HOME/.freesurfer/license.txt"
```

The file `license_freesurfer.txt.example` is a placeholder template only.
