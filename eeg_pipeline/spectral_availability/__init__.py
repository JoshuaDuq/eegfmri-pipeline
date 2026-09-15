from eeg_pipeline.spectral_availability.estimators import (
    morlet_half_support,
    morlet_temporal_half_support,
    multitaper_half_support,
    multitaper_tfr_half_support,
    welch_half_support,
)
from eeg_pipeline.spectral_availability.model import (
    EpochSpectralAvailability,
    FrequencyInterval,
    RecordingExclusions,
    RecordingKey,
    merge_frequency_intervals,
)


def __getattr__(name: str):
    if name in {"DecombManifest", "load_decomb_manifest"}:
        from eeg_pipeline.spectral_availability.decomb import (
            DecombManifest,
            load_decomb_manifest,
        )

        exports = {
            "DecombManifest": DecombManifest,
            "load_decomb_manifest": load_decomb_manifest,
        }
    elif name == "align_decomb_to_epochs":
        from eeg_pipeline.spectral_availability.alignment import (
            align_decomb_to_epochs,
        )

        exports = {"align_decomb_to_epochs": align_decomb_to_epochs}
    elif name in {"AvailabilityAuditRow", "SpectralAvailabilityAudit"}:
        from eeg_pipeline.spectral_availability.audit import (
            AvailabilityAuditRow,
            SpectralAvailabilityAudit,
        )

        exports = {
            "AvailabilityAuditRow": AvailabilityAuditRow,
            "SpectralAvailabilityAudit": SpectralAvailabilityAudit,
        }
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = exports[name]
    globals()[name] = value
    return value


__all__ = [
    "AvailabilityAuditRow",
    "DecombManifest",
    "EpochSpectralAvailability",
    "FrequencyInterval",
    "RecordingExclusions",
    "RecordingKey",
    "SpectralAvailabilityAudit",
    "align_decomb_to_epochs",
    "load_decomb_manifest",
    "morlet_half_support",
    "morlet_temporal_half_support",
    "merge_frequency_intervals",
    "multitaper_half_support",
    "multitaper_tfr_half_support",
    "welch_half_support",
]
