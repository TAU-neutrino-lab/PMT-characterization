"""Versioned provenance for Selection.ipynb dataframe caches."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


SELECTION_CACHE_SCHEMA_VERSION = 3

REQUIRED_PREPROCESSING_SETTINGS = frozenset(
    {
        "channel",
        "baseline_window_ns",
        "led_time_ns",
        "pre_led_ns",
        "post_led_ns",
        "peak_snr_threshold",
        "peak_prominence_snr",
        "peak_distance_samples",
        "peak_width_samples",
        "require_clean_baseline",
        "baseline_clean_snr",
        "baseline_reference_path",
        "skip_corrupt_files",
        "fixed_pulse_window_charge",
    }
)

REQUIRED_SELECTION_SETTINGS = frozenset(
    {
        "cut_thresholds_snr",
        "selection_mode",
        "timing_reference_snr",
        "peak_timing_tolerance_ns",
        "timing_reference_requires_single_peak",
        "max_allowed_peaks",
        "include_no_peak_cuts",
        "include_shape_cut",
        "min_shape_reference_pulses",
        "shape_quantiles",
        "selection_names",
        "generated_selection_names",
    }
)

REQUIRED_FIXED_PULSE_WINDOW_SETTINGS = frozenset(
    {
        "enabled",
        "column",
        "reference_snr",
        "width_quantile",
        "min_reference_pulses",
        "read_chunk_size",
        "reference_pulses",
        "center_ns",
        "width_ns",
        "window_ns",
        "reference_containment_fraction",
    }
)


def _require_settings(settings, required_keys, section):
    if not isinstance(settings, dict):
        raise ValueError(f"Selection cache {section} must be a dictionary")
    missing = required_keys.difference(settings)
    if missing:
        raise ValueError(
            f"Selection cache {section} is missing settings: {sorted(missing)}"
        )


def build_selection_cache_provenance(
    *,
    acquisition,
    data_dir,
    source_files,
    preprocessing,
    selection,
):
    """Build the complete manifest stored in a pre-cut dataframe's attrs."""

    _require_settings(
        preprocessing, REQUIRED_PREPROCESSING_SETTINGS, "preprocessing manifest"
    )
    _require_settings(
        preprocessing["fixed_pulse_window_charge"],
        REQUIRED_FIXED_PULSE_WINDOW_SETTINGS,
        "fixed pulse-window charge manifest",
    )
    _require_settings(selection, REQUIRED_SELECTION_SETTINGS, "selection manifest")
    return {
        "schema_version": SELECTION_CACHE_SCHEMA_VERSION,
        "acquisition": str(acquisition),
        "data_dir": str(Path(data_dir).resolve()),
        "source_files": [str(Path(path).resolve()) for path in source_files],
        "preprocessing": deepcopy(preprocessing),
        "selection": deepcopy(selection),
    }


def load_selection_cache_provenance(dataframe, *, expected_acquisition=None):
    """Return a validated manifest or fail closed for stale/legacy caches."""

    provenance = dataframe.attrs.get("pmt_selection")
    if not isinstance(provenance, dict):
        raise ValueError(
            "Cached dataframe has no Selection manifest. Rerun Selection.ipynb "
            "before fitting."
        )
    schema_version = provenance.get("schema_version")
    if schema_version != SELECTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "Cached dataframe uses Selection manifest schema "
            f"{schema_version!r}; expected {SELECTION_CACHE_SCHEMA_VERSION}. "
            "Rerun Selection.ipynb before fitting."
        )

    required_top_level = {
        "acquisition",
        "data_dir",
        "source_files",
        "preprocessing",
        "selection",
    }
    missing = required_top_level.difference(provenance)
    if missing:
        raise ValueError(
            f"Selection cache manifest is missing fields: {sorted(missing)}. "
            "Rerun Selection.ipynb before fitting."
        )
    if expected_acquisition is not None and provenance["acquisition"] != str(
        expected_acquisition
    ):
        raise ValueError(
            "Cached acquisition does not match the requested file_name: "
            f"cache={provenance['acquisition']!r}, "
            f"requested={str(expected_acquisition)!r}."
        )
    if not isinstance(provenance["source_files"], (list, tuple)) or not provenance[
        "source_files"
    ]:
        raise ValueError("Selection cache manifest contains no source HDF5 files")

    _require_settings(
        provenance["preprocessing"],
        REQUIRED_PREPROCESSING_SETTINGS,
        "preprocessing manifest",
    )
    _require_settings(
        provenance["preprocessing"]["fixed_pulse_window_charge"],
        REQUIRED_FIXED_PULSE_WINDOW_SETTINGS,
        "fixed pulse-window charge manifest",
    )
    _require_settings(
        provenance["selection"],
        REQUIRED_SELECTION_SETTINGS,
        "selection manifest",
    )
    return deepcopy(provenance)


__all__ = [
    "SELECTION_CACHE_SCHEMA_VERSION",
    "build_selection_cache_provenance",
    "load_selection_cache_provenance",
]
