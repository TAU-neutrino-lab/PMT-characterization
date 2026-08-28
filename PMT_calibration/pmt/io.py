"""PMT waveform loading helpers.

This module is a PMT-facing layer over :mod:`lab_tools.io`. Keep generic
oscilloscope file parsing in ``lab_tools``; put PMT workflow conveniences here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import gc
import json
import re
import time
import warnings
import pandas as pd
import numpy as np


from lab_tools.io import (
    iter_keysight_chunks,
    read_keysight_h5_direct,
    read_segment_time_tags,
    standard_units,
)

from .preprocessing import *
from .selection import *


_PMT_VOLTAGE_RE = re.compile(r"(?:^|_)(?P<voltage>\d+(?:\.\d+)?)V(?:_|$)")


def _acquisition_stem(name: str | Path) -> str:
    """Remove only known data-artifact suffixes, preserving decimal voltages."""

    filename = Path(name).name
    for suffix in (".h5", ".npz"):
        if filename.lower().endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def extract_pmt_voltage(name: str | Path) -> float:
    """Extract the PMT bias voltage from an acquisition or artifact name."""

    match = _PMT_VOLTAGE_RE.search(_acquisition_stem(name))
    if match is None:
        raise ValueError(f"Could not find a '<voltage>V' token in {name!s}")
    return float(match.group("voltage"))


def load_acquisition_timing_exposure(
    source_files,
    *,
    acquisition=None,
    timing_csv_path=None,
    time_column="Acquisition_s",
):
    """Load live exposure for the HDF5 files represented by an acquisition.

    The timing CSV is discovered beside the source HDF5 files unless an
    explicit path is supplied. Only rows matching ``source_files`` contribute,
    and every requested source file must have exactly one timing row.
    """

    source_paths = [Path(path) for path in source_files]
    if not source_paths:
        return None
    source_names = {path.name for path in source_paths}
    if len(source_names) != len(source_paths):
        raise ValueError("Timing exposure received duplicate source filenames")

    if timing_csv_path is None:
        candidates = []
        if acquisition is not None:
            timing_name = f"{acquisition}_timing.csv"
            candidates.extend(
                path.parent / timing_name for path in source_paths
            )
        candidates = list(dict.fromkeys(candidates))
        timing_paths = [path for path in candidates if path.is_file()]
        if not timing_paths:
            return None
        resolved_paths = {path.resolve() for path in timing_paths}
        if len(resolved_paths) != 1:
            raise ValueError(
                f"Source files resolve multiple timing CSVs: {sorted(map(str, resolved_paths))}"
            )
        timing_csv_path = timing_paths[0]
    else:
        timing_csv_path = Path(timing_csv_path)
        if not timing_csv_path.is_file():
            raise FileNotFoundError(f"Timing CSV does not exist: {timing_csv_path}")

    timing = pd.read_csv(timing_csv_path, encoding="utf-8-sig")
    required_columns = {"Filename", "Segments", time_column}
    missing_columns = required_columns.difference(timing.columns)
    if missing_columns:
        raise ValueError(
            f"Timing CSV {timing_csv_path} is missing columns "
            f"{sorted(missing_columns)}"
        )
    if timing["Filename"].duplicated().any():
        duplicates = timing.loc[
            timing["Filename"].duplicated(keep=False), "Filename"
        ].tolist()
        raise ValueError(
            f"Timing CSV {timing_csv_path} repeats filenames: {duplicates}"
        )
    timing = timing.loc[timing["Filename"].isin(source_names)].copy()
    logged_names = set(timing["Filename"])
    missing_files = sorted(source_names.difference(logged_names))
    if missing_files:
        raise ValueError(
            f"Timing CSV {timing_csv_path} has no rows for {missing_files}"
        )
    timing["Segments"] = pd.to_numeric(timing["Segments"], errors="raise")
    timing[time_column] = pd.to_numeric(timing[time_column], errors="raise")
    if (timing["Segments"] < 0).any() or (timing[time_column] <= 0).any():
        raise ValueError(
            f"Timing CSV {timing_csv_path} contains invalid segments or {time_column}"
        )
    live_time_s = float(timing[time_column].sum())
    logged_segments = int(timing["Segments"].sum())
    return {
        "timing_csv_path": Path(timing_csv_path),
        "time_column": str(time_column),
        "live_time_s": live_time_s,
        "logged_segments": logged_segments,
        "logged_rate_evt_s": logged_segments / live_time_s,
        "n_files": len(timing),
    }


def resolve_baseline_reference_path(
    reference: str | Path | None,
    acquisition_name: str | Path,
) -> Path | None:
    """Resolve a fixed or per-voltage median-baseline reference.

    ``reference`` may be a single ``.npz`` file (the legacy behavior) or a
    directory containing one artifact per voltage.  Within a directory, the
    preferred names are ``<acquisition>.npz``, ``<PMT>_<voltage>V.npz``, and
    ``<voltage>V.npz``.  As a fallback, exactly one artifact containing the
    same voltage token is accepted.
    """

    if reference is None:
        return None

    reference = Path(reference)
    if reference.is_file():
        if reference.suffix.lower() != ".npz":
            raise ValueError(f"Baseline reference must be an .npz file: {reference}")
        return reference
    if not reference.exists():
        raise FileNotFoundError(f"Missing baseline-reference path: {reference}")
    if not reference.is_dir():
        raise ValueError(f"Baseline reference is not a file or directory: {reference}")

    acquisition_stem = _acquisition_stem(acquisition_name)
    voltage_match = _PMT_VOLTAGE_RE.search(acquisition_stem)
    if voltage_match is None:
        raise ValueError(
            f"Cannot select a voltage-specific baseline reference for {acquisition_name!s}"
        )
    voltage_token = f"{voltage_match.group('voltage')}V"
    pmt_prefix = acquisition_stem[: voltage_match.start()].rstrip("_")
    preferred_names = [
        f"{acquisition_stem}.npz",
        f"{pmt_prefix}_{voltage_token}.npz" if pmt_prefix else "",
        f"{voltage_token}.npz",
    ]
    for filename in preferred_names:
        if filename and (reference / filename).is_file():
            return reference / filename

    target_voltage = float(voltage_match.group("voltage"))
    voltage_matches = []
    for candidate in sorted(reference.glob("*.npz")):
        try:
            candidate_voltage = extract_pmt_voltage(candidate)
        except ValueError:
            continue
        if np.isclose(candidate_voltage, target_voltage, rtol=0.0, atol=1e-9):
            voltage_matches.append(candidate)

    if len(voltage_matches) == 1:
        return voltage_matches[0]
    if not voltage_matches:
        raise FileNotFoundError(
            f"No {voltage_token} median baseline reference found in {reference}. "
            f"Expected a file such as {reference / f'{voltage_token}.npz'}"
        )
    matches = ", ".join(str(path) for path in voltage_matches)
    raise ValueError(
        f"Multiple {voltage_token} baseline references found in {reference}: {matches}. "
        "Use one unambiguous preferred filename or a more specific directory."
    )


def load_baseline_reference(path: str | Path) -> dict:
    """Load and validate a median-baseline ``.npz`` artifact."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as reference_data:
        missing = {"time_ns", "baseline_template_mV"}.difference(reference_data.files)
        if missing:
            raise ValueError(
                f"Baseline reference {path} is missing arrays: {sorted(missing)}"
            )
        time_ns = np.asarray(reference_data["time_ns"], dtype=float).copy()
        template_mV = np.asarray(
            reference_data["baseline_template_mV"], dtype=float
        ).copy()
        metadata = {}
        if "metadata_json" in reference_data.files:
            metadata = json.loads(str(reference_data["metadata_json"].item()))

    if time_ns.ndim != 1 or template_mV.ndim != 1:
        raise ValueError(f"Baseline reference arrays must be one-dimensional: {path}")
    if time_ns.shape != template_mV.shape:
        raise ValueError(f"Baseline reference time/template shapes do not match: {path}")
    if not np.all(np.isfinite(time_ns)) or not np.all(np.isfinite(template_mV)):
        raise ValueError(f"Baseline reference contains non-finite values: {path}")
    return {
        "path": path,
        "time_ns": time_ns,
        "baseline_template_mV": template_mV,
        "metadata": metadata,
    }


def find_pmt_files(
    data_dir: str | Path,
    file_name: str,
    *,
    max_files: int | None = None,
) -> list[Path]:
    """Return sorted PMT HDF5 files matching ``file_name-*.h5``."""

    files = sorted(Path(data_dir).glob(f"{file_name}-*.h5"))
    if max_files is not None:
        files = files[:max_files]
    return files

def read_waveform_sample(
    files: str | Path | Sequence[str | Path],
    *,
    max_events: int | None = 1000,
    channel: str = "Channel 1",
    time_origin="original" # options "original", "zero"
):
    """Read a sample of segmented PMT waveforms from one or more files.

    Returns a dictionary with ``time_ns``, ``voltage_mV``, ``metadata``, and
    bookkeeping fields.
    """

    if isinstance(files, (str, Path)):
        files = [files]

    all_time = []
    all_voltage = []
    all_metadata = []
    all_segments = []

    remaining = max_events

    for file in files:
        available_segments, _ = read_segment_time_tags(file, channel=channel)

        if remaining is None:
            sample_segments = available_segments
        else:
            sample_segments = available_segments[:remaining]

        if len(sample_segments) == 0:
            continue

        time_s, voltage_V, metadata = read_keysight_h5_direct(
            file,
            channel=channel,
            segment_numbers=sample_segments,
        )

        time_ns, voltage_mV, adc_step_mV, units_time, units_voltage = standard_units(
            time_s,
            voltage_V,
            metadata,
        )

        all_time.append(time_ns)
        all_voltage.append(voltage_mV)
        all_metadata.append(metadata)
        all_segments.extend(sample_segments)

        if remaining is not None:
            remaining -= len(sample_segments)
            if remaining <= 0:
                break
        
    time_ns = np.concatenate(all_time, axis=0)
    if time_origin=="zero":
        time_ns = time_ns - time_ns[0]

    return {
        "time_ns": time_ns,
        "voltage_mV": np.concatenate(all_voltage, axis=0),
        "metadata": all_metadata,
        "sample_segments": all_segments,
        "adc_step_mV": adc_step_mV,
        "units_time": units_time,
        "units_voltage": units_voltage,
    }

def load_preprocessed_waveforms(
    files,
    *,
    channel="Channel 1",
    chunk_size=512,
    remove_saturation=True,
    subtract_baseline=True,
    baseline_window_ns=(0.0, 20.0),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    low_limit_mV=None,
    high_limit_mV=None,
    margin_mV=0.0,
    max_saturated_samples=0,
    time_origin="original" # options "original", "zero"
):
    """
    Load waveforms and optionally apply preprocessing.

    Processing order:

        raw waveforms
            ↓
        saturation rejection
            ↓
        baseline subtraction

    Returns
    -------
    dict
        Contains waveform arrays and event-level metadata.
    """

    voltage_parts = []
    event_files = []
    event_segments = []

    event_info_parts = {}

    time_ns = None

    total_events = 0
    total_kept = 0
    total_removed = 0

    for chunk in iter_keysight_chunks( files, channel=channel, chunk_size=chunk_size ):

        if time_ns is None:
            time_ns_original = chunk["time_ns"]
            if time_origin == "original":
                time_ns = time_ns_original
            elif time_origin == "zero":
                time_ns = time_ns_original - time_ns_original[0]
            else:
                raise ValueError("time_origin must be 'original' or 'zero'")

        elif len(time_ns_original) != len(chunk["time_ns"]):
            raise ValueError( f"Number of samples changed in {chunk['filename']}")
        # elif not np.allclose(chunk["time_ns"], time_ns_original): # checks whether two arrays are element-wise approximately equal within a given mathematical tolerance
        #     print(chunk["time_ns"])
        #     print()
        #     print(time_ns_original)
        #     raise ValueError( f"Time axis changed in {chunk['filename']}")

        voltage_mV = chunk["voltage_mV"]
        n_events_chunk = len(voltage_mV)

        # ---------- Saturation Rejection ----------

        if remove_saturation:
            voltage_mV, keep_mask, saturation_event_info, saturation_summary = (
                remove_saturated_waveforms(
                    voltage_mV,
                    chunk["metadata"],
                    low_limit_mV=low_limit_mV,
                    high_limit_mV=high_limit_mV,
                    margin_mV=margin_mV,
                    max_saturated_samples=max_saturated_samples,
                )
            )

            # accumulate statistics over all chunks
            total_events += saturation_summary["n_total"]
            total_kept += saturation_summary["n_kept"]
            total_removed += saturation_summary["n_removed"]

            segment_numbers = np.asarray( chunk["segment_numbers"])[keep_mask]
            event_files_chunk = np.array( [chunk["filename"]] * n_events_chunk)[keep_mask]

        else:
            keep_mask = np.ones( n_events_chunk, dtype=bool)
            segment_numbers = np.asarray( chunk["segment_numbers"] )
            event_files_chunk = np.array( [chunk["filename"]] * n_events_chunk)
            saturation_event_info = {}

        # ---------- Baseline subtraction ----------

        if subtract_baseline:
            voltage_mV, baseline_event_info = baseline_subtraction(
                time_ns,
                voltage_mV,
                baseline_window_ns=baseline_window_ns,
                baseline_reference_time_ns=baseline_reference_time_ns,
                baseline_reference_mV=baseline_reference_mV,
            )
        else:
            baseline_event_info = {}

        # ---------- Merge event-level metadata ----------

        chunk_event_info = {
            **saturation_event_info,
            **baseline_event_info,
        }

        for key, values in chunk_event_info.items():
            if key not in event_info_parts:
                event_info_parts[key] = []

            event_info_parts[key].append( np.asarray(values))

        
        # ---------- Store waveforms ----------

        voltage_parts.append( voltage_mV)
        event_files.extend( event_files_chunk)
        event_segments.extend( segment_numbers)

    if time_ns is None:
        raise ValueError( "No waveforms were loaded")

    # ---------- Concatenate event metadata ----------

    event_info = {}

    for key, values in event_info_parts.items():
        event_info[key] = np.concatenate( values )

    result = {
        "time_ns":       time_ns,
        "voltage_mV":    np.concatenate( voltage_parts, axis=0 ),
        "event_file":    np.asarray( event_files ), # from which file this specific event was extracted
        "event_segment": np.asarray( event_segments ),
        "event_info":    event_info,
    }

    if remove_saturation:
        result["saturation_summary"] = {
            "n_total": total_events,
            "n_kept": total_kept,
            "n_removed": total_removed,
            "efficiency": ( total_kept / total_events if total_events > 0 else np.nan )
        }

    return result

def load_files_streaming(
    files,
    *,
    channel="Channel 1",
    chunk_size=512,
    baseline_window_ns=(0, 20),
    time_origin="zero",
    keep_waveform_sample=0,
    led_time_ns=None,
    pre_led_ns=20.0,
    post_led_ns=80.0,
    peak_snr_threshold=5.0,
    peak_prominence_snr=None,
    peak_distance_samples=None,
    peak_width_samples=None,
    baseline_rms_max_quantile=None,
    require_clean_baseline=False,
    baseline_clean_snr=8.0,
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    progress_interval_s=10.0,
    skip_corrupt_files=True,
):
    """Load and reduce waveform chunks without retaining all waveforms.

    Progress is reported at most once per ``progress_interval_s`` seconds,
    plus one final update. Set the interval to ``None`` to disable progress.

    When ``baseline_rms_max_quantile`` is supplied, a lightweight first pass
    learns that quantile of the baseline-window RMS distribution after the
    saturation cut. Waveforms above the learned threshold are rejected before
    peak finding and feature extraction. Only scalar RMS values are retained
    during this pass, never waveform arrays.

    When ``skip_corrupt_files`` is true, HDF5 read errors skip the affected
    acquisition segment with an explicit warning while processing continues.
    """
    files = list(files)
    df_parts = []
    waveform_sample_parts = []
    time_ns_ref = None

    total_events = 0
    total_kept_after_saturation = 0
    total_kept_after_baseline_rms_cut = 0
    total_kept_after_raw_baseline_check = 0
    total_kept_after_baseline_cut = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    current_file = None
    files_started = 0
    skipped_files = []

    baseline_rms_threshold_mV = None
    baseline_rms_learning_count = 0
    if baseline_rms_max_quantile is not None:
        baseline_rms_max_quantile = float(baseline_rms_max_quantile)
        if not 0.0 < baseline_rms_max_quantile <= 1.0:
            raise ValueError("baseline_rms_max_quantile must be in (0, 1]")
        learned_rms_parts = []
        learning_started_at = time.monotonic()
        print(
            "Learning baseline-window RMS cutoff at quantile "
            f"{baseline_rms_max_quantile:g}..."
        )
        for file in files:
            try:
                for chunk in iter_keysight_chunks(
                    [file], channel=channel, chunk_size=chunk_size
                ):
                    learning_time_ns = np.asarray(chunk["time_ns"], dtype=float)
                    if time_origin == "zero":
                        learning_time_ns = learning_time_ns - learning_time_ns[0]
                    baseline_mask = (
                        (learning_time_ns >= baseline_window_ns[0])
                        & (learning_time_ns <= baseline_window_ns[1])
                    )
                    if not np.any(baseline_mask):
                        raise ValueError(
                            f"Baseline window {baseline_window_ns} contains no samples"
                        )
                    learning_waveforms, _, _, _ = remove_saturated_waveforms(
                        chunk["voltage_mV"],
                        chunk["metadata"],
                        low_limit_mV=None,
                        high_limit_mV=None,
                        margin_mV=0.0,
                        max_saturated_samples=0,
                    )
                    learned_rms_parts.append(
                        np.std(learning_waveforms[:, baseline_mask], axis=1)
                    )
            except (OSError, RuntimeError) as exc:
                if not skip_corrupt_files:
                    raise
                if str(file) not in skipped_files:
                    skipped_files.append(str(file))
                warnings.warn(
                    f"Skipping unreadable HDF5 segment {file}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not learned_rms_parts:
            raise ValueError("No baseline RMS values were available to learn the cutoff")
        learned_rms_values = np.concatenate(learned_rms_parts)
        learned_rms_values = learned_rms_values[np.isfinite(learned_rms_values)]
        if not len(learned_rms_values):
            raise ValueError("All learned baseline RMS values are non-finite")
        baseline_rms_learning_count = len(learned_rms_values)
        baseline_rms_threshold_mV = float(
            np.quantile(learned_rms_values, baseline_rms_max_quantile)
        )
        print(
            f"Learned baseline RMS threshold: {baseline_rms_threshold_mV:.6g} mV "
            f"from {baseline_rms_learning_count:,} non-saturated waveforms "
            f"({time.monotonic() - learning_started_at:.1f} s)."
        )
        del learned_rms_values, learned_rms_parts

    def iter_readable_chunks():
        # Iterate one file at a time so an HDF5 failure does not terminate the
        # generator before the remaining acquisition segments are attempted.
        for file in files:
            try:
                yield from iter_keysight_chunks(
                    [file], channel=channel, chunk_size=chunk_size
                )
            except (OSError, RuntimeError) as exc:
                if not skip_corrupt_files:
                    raise
                if str(file) not in skipped_files:
                    skipped_files.append(str(file))
                warnings.warn(
                    f"Skipping unreadable HDF5 segment {file}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    for chunk in iter_readable_chunks():
        chunk_file = str(chunk["filename"])
        if chunk_file != current_file:
            current_file = chunk_file
            files_started += 1
        time_ns = chunk["time_ns"]

        if time_origin == "zero":
            time_ns = time_ns - time_ns[0]

        if time_ns_ref is None:
            time_ns_ref = time_ns
        elif len(time_ns_ref) != len(time_ns):
            raise ValueError(f"Number of samples changed in {chunk['filename']}")

        voltage_mV = chunk["voltage_mV"]
        n_events_chunk = len(voltage_mV)
        total_events += n_events_chunk

        # 1. Remove saturated events
        voltage_mV, keep_mask, saturation_event_info, saturation_summary = (
            remove_saturated_waveforms(
                voltage_mV,
                chunk["metadata"],
                low_limit_mV=None,
                high_limit_mV=None,
                margin_mV=0.0,
                max_saturated_samples=0,
            )
        )

        total_kept_after_saturation += len(voltage_mV)

        segment_numbers = np.asarray(chunk["segment_numbers"])[keep_mask]
        event_files_chunk = np.array([chunk["filename"]] * n_events_chunk)[keep_mask]

        # Reject the globally noisiest baseline windows before later analysis.
        if baseline_rms_threshold_mV is not None:
            baseline_mask = (
                (time_ns_ref >= baseline_window_ns[0])
                & (time_ns_ref <= baseline_window_ns[1])
            )
            chunk_baseline_rms = np.std(voltage_mV[:, baseline_mask], axis=1)
            baseline_rms_keep_mask = (
                np.isfinite(chunk_baseline_rms)
                & (chunk_baseline_rms <= baseline_rms_threshold_mV)
            )
            voltage_mV = voltage_mV[baseline_rms_keep_mask]
            segment_numbers = segment_numbers[baseline_rms_keep_mask]
            event_files_chunk = event_files_chunk[baseline_rms_keep_mask]
        total_kept_after_baseline_rms_cut += len(voltage_mV)

        # 2. Reject raw baseline windows containing a pulse-like excursion. This
        # must happen before subtraction so the pulse cannot bias its own noise
        # threshold. It is especially important for asynchronous dark counts.
        if require_clean_baseline:
            baseline_keep_mask, raw_baseline_info = baseline_cleanliness_mask(
                time_ns_ref,
                voltage_mV,
                baseline_window_ns=baseline_window_ns,
                excursion_snr=baseline_clean_snr,
                polarity="negative",
            )
            voltage_mV = voltage_mV[baseline_keep_mask]
            segment_numbers = segment_numbers[baseline_keep_mask]
            event_files_chunk = event_files_chunk[baseline_keep_mask]
            raw_baseline_info = {
                key: np.asarray(values)[baseline_keep_mask]
                for key, values in raw_baseline_info.items()
            }
        else:
            raw_baseline_info = {}
        total_kept_after_raw_baseline_check += len(voltage_mV)

        # 3. Baseline subtraction
        voltage_mV, baseline_event_info = baseline_subtraction(
            time_ns_ref,
            voltage_mV,
            baseline_window_ns=baseline_window_ns,
            baseline_reference_time_ns=baseline_reference_time_ns,
            baseline_reference_mV=baseline_reference_mV,
        )

        # 4. Build features only for this chunk
        extra = {
            **raw_baseline_info,
            **baseline_event_info,
            "event_file": event_files_chunk,
            "event_segment": segment_numbers,
        }

        df_chunk = build_waveform_feature_dataframe(
            time_ns_ref,
            voltage_mV,
            peak_threshold_mV=None,
            peak_snr_threshold=peak_snr_threshold,
            peak_prominence_snr=peak_prominence_snr,
            peak_distance_samples=peak_distance_samples,
            peak_width_samples=peak_width_samples,
            pre_peak_ns=20,
            post_peak_ns=80,
            pre_rise_ns=10,
            post_rise_ns=80,
            led_time_ns=led_time_ns,
            pre_led_ns=pre_led_ns,
            post_led_ns=post_led_ns,
            extra=extra,
        )

        # 5. Apply the existing post-subtraction baseline-pulse safeguard.
        mask = reject_baseline_pulses(
            df_chunk,
            baseline_window_ns,
            min_peak_height_mV=0.5,
        )

        df_chunk_sel = df_chunk[mask].copy()
        total_kept_after_baseline_cut += len(df_chunk_sel)

        df_parts.append(df_chunk_sel)

        # Optional: keep only a small waveform sample for plotting
        if keep_waveform_sample > 0:
            remaining = keep_waveform_sample - sum(len(x) for x in waveform_sample_parts)
            if remaining > 0:
                waveform_sample_parts.append(voltage_mV[mask.values][:remaining].copy())

        # 6. Explicitly release chunk arrays
        del voltage_mV, df_chunk, df_chunk_sel
        gc.collect()

        now = time.monotonic()
        if (
            progress_interval_s is not None
            and now - last_progress_at >= progress_interval_s
        ):
            elapsed_s = now - started_at
            print(
                f"Loading: {files_started}/{len(files)} files started, "
                f"{total_events:,} events processed, "
                f"{total_kept_after_baseline_cut:,} kept "
                f"({elapsed_s:.0f} s elapsed)"
            )
            last_progress_at = now

    if not df_parts:
        skipped_note = (
            f"; skipped {len(skipped_files)} unreadable file(s)"
            if skipped_files
            else ""
        )
        raise ValueError(f"No waveforms were loaded{skipped_note}")

    df_sel = pd.concat(df_parts, ignore_index=True)

    elapsed_s = time.monotonic() - started_at
    print(
        f"Loading complete: {len(files) - len(skipped_files)}/{len(files)} files, "
        f"{total_events:,} events in {elapsed_s:.1f} s"
    )
    if skipped_files:
        print(f"Skipped unreadable files: {len(skipped_files)}")
        for skipped_file in skipped_files:
            print(f"  - {skipped_file}")
    print(f"Initial events: {total_events}")
    print(f"After saturation cut: {total_kept_after_saturation}")
    if baseline_rms_threshold_mV is not None:
        print(
            f"After baseline RMS quantile cut "
            f"(<= {baseline_rms_threshold_mV:.6g} mV, "
            f"q={baseline_rms_max_quantile:g}): "
            f"{total_kept_after_baseline_rms_cut}"
        )
    if require_clean_baseline:
        print(
            f"After raw baseline-cleanliness cut "
            f"(< {baseline_clean_snr:g} robust RMS): "
            f"{total_kept_after_raw_baseline_check}"
        )
    print(f"After baseline-pulse cut: {total_kept_after_baseline_cut}")

    if keep_waveform_sample > 0 and waveform_sample_parts:
        waveforms_sample = np.concatenate(waveform_sample_parts, axis=0)
    else:
        waveforms_sample = None

    df_sel.attrs["baseline_rms_quantile_cut"] = {
        "enabled": baseline_rms_threshold_mV is not None,
        "quantile": baseline_rms_max_quantile,
        "threshold_mV": baseline_rms_threshold_mV,
        "learning_waveforms": baseline_rms_learning_count,
        "kept_after_saturation": total_kept_after_saturation,
        "kept_after_rms_cut": total_kept_after_baseline_rms_cut,
    }
    return time_ns_ref, df_sel, waveforms_sample

def load_files_one_go(
    files,
    chunk_size,
    baseline_window_ns,
    channel,
    *,
    led_time_ns=None,
    pre_led_ns=20.0,
    post_led_ns=80.0,
    peak_snr_threshold=5.0,
    peak_prominence_snr=None,
    peak_distance_samples=None,
    peak_width_samples=None,
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
):
    prerocessed_data = load_preprocessed_waveforms(
    files,
    chunk_size=chunk_size,
    remove_saturation=True,
    subtract_baseline=True,
    baseline_window_ns=baseline_window_ns,
    low_limit_mV=None, 
    high_limit_mV=None, 
    margin_mV=0.0,
    max_saturated_samples=0,
    channel=channel,
    time_origin='zero',
    baseline_reference_time_ns=baseline_reference_time_ns,
    baseline_reference_mV=baseline_reference_mV,
    )
    time_ns = prerocessed_data["time_ns"]
    voltage_mV = prerocessed_data["voltage_mV"]

    preprocessed_event_info = {k: v for k, v in prerocessed_data['event_info'].items() if k not in ['is_saturated', 'n_saturated_samples']}
    preprocessed_event_info.update({ 
        'event_file': prerocessed_data['event_file'],
        'event_segment': prerocessed_data['event_segment']
    })

    df = build_waveform_feature_dataframe(
        time_ns,
        voltage_mV,
        peak_threshold_mV=None,
        peak_snr_threshold=peak_snr_threshold,
        peak_prominence_snr=peak_prominence_snr,
        peak_distance_samples=peak_distance_samples,
        peak_width_samples=peak_width_samples,
        pre_peak_ns = 20,
        post_peak_ns = 80,
        pre_rise_ns = 10, 
        post_rise_ns = 80, 
        led_time_ns=led_time_ns,
        pre_led_ns=pre_led_ns,
        post_led_ns=post_led_ns,
        extra=preprocessed_event_info 
    )
    cutflow = CutFlow(len(df))

    cutflow.apply( "peaks in the baseline", reject_baseline_pulses( df, baseline_window_ns, min_peak_height_mV=0.5 ) )
    cutflow.print()

    df_sel = df[cutflow.mask].copy()
    waveforms_sel = voltage_mV[cutflow.mask]

    return time_ns, waveforms_sel, df_sel


def integrate_event_fixed_window_charge(
    events,
    *,
    window_ns,
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    read_chunk_size=512,
    progress_interval_s=10.0,
    time_origin="zero",
    polarity="negative",
):
    """Integrate one trigger-relative window for an event dataframe.

    Source files are scanned sequentially in bounded chunks and only retained
    ``event_file``/``event_segment`` rows are integrated.  Sequential scanning
    is substantially faster than HDF5 fancy indexing after event cuts make the
    requested segment numbers non-contiguous. The returned charge array
    preserves dataframe row order. Every event receives exactly the same time
    bounds, including events with no detected peak.
    """

    if len(events) == 0:
        raise ValueError("No events were requested")
    missing = {"event_file", "event_segment"}.difference(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if len(window_ns) != 2:
        raise ValueError("window_ns must contain (start_ns, stop_ns)")
    start_ns, stop_ns = (float(value) for value in window_ns)
    if not np.isfinite(start_ns) or not np.isfinite(stop_ns) or start_ns >= stop_ns:
        raise ValueError(f"Invalid fixed charge window: {window_ns!r}")
    read_chunk_size = int(read_chunk_size)
    if read_chunk_size < 1:
        raise ValueError("read_chunk_size must be at least 1")
    if progress_interval_s is not None:
        progress_interval_s = float(progress_interval_s)
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s must be positive or None")

    event_table = events.loc[:, ["event_file", "event_segment"]].copy()
    event_table["_request_order"] = np.arange(len(event_table))
    charges_mV_ns = np.full(len(event_table), np.nan)
    time_ns_ref = None
    center_ns = 0.5 * (start_ns + stop_ns)
    half_width_ns = 0.5 * (stop_ns - start_ns)

    grouped_events = list(event_table.groupby("event_file", sort=False))
    started_at = time.monotonic()
    last_progress_at = started_at
    integrated_events = 0
    scanned_events = 0

    for file_index, (event_file, group) in enumerate(grouped_events, start=1):
        segment_numbers = group["event_segment"].to_numpy(dtype=int)
        if len(np.unique(segment_numbers)) != len(segment_numbers):
            raise ValueError(f"Duplicate event segments requested from {event_file}")
        request_order_by_segment = dict(zip(
            segment_numbers,
            group["_request_order"].to_numpy(dtype=int),
        ))
        found_segments = set()

        for chunk in iter_keysight_chunks(
            [event_file], channel=channel, chunk_size=read_chunk_size
        ):
            time_ns = np.asarray(chunk["time_ns"], dtype=float)
            if time_origin == "zero":
                time_ns = time_ns - time_ns[0]
            elif time_origin != "original":
                raise ValueError("time_origin must be 'original' or 'zero'")

            if time_ns_ref is None:
                time_ns_ref = time_ns
                if start_ns < time_ns[0] or stop_ns > time_ns[-1]:
                    raise ValueError(
                        f"Fixed charge window [{start_ns:g}, {stop_ns:g}] ns is "
                        f"not contained in waveform [{time_ns[0]:g}, {time_ns[-1]:g}] ns"
                    )
            elif not np.allclose(time_ns_ref, time_ns, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Time axis changed in {event_file}")

            chunk_segments = np.asarray(chunk["segment_numbers"], dtype=int)
            scanned_events += len(chunk_segments)
            selected_positions = np.fromiter(
                (
                    position
                    for position, segment in enumerate(chunk_segments)
                    if segment in request_order_by_segment
                ),
                dtype=int,
            )
            if selected_positions.size:
                selected_segments = chunk_segments[selected_positions]
                voltage_mV = np.asarray(chunk["voltage_mV"])[selected_positions]
                voltage_mV, _ = baseline_subtraction(
                    time_ns,
                    voltage_mV,
                    baseline_window_ns=baseline_window_ns,
                    baseline_reference_time_ns=baseline_reference_time_ns,
                    baseline_reference_mV=baseline_reference_mV,
                )
                chunk_charges = integrate_led_window_charge(
                    time_ns,
                    voltage_mV,
                    led_time_ns=center_ns,
                    pre_led_ns=half_width_ns,
                    post_led_ns=half_width_ns,
                    polarity=polarity,
                    require_full_window=True,
                )
                request_orders = np.fromiter(
                    (
                        request_order_by_segment[int(segment)]
                        for segment in selected_segments
                    ),
                    dtype=int,
                )
                charges_mV_ns[request_orders] = chunk_charges
                found_segments.update(int(segment) for segment in selected_segments)
                integrated_events += len(selected_segments)

            now = time.monotonic()
            if (
                progress_interval_s is not None
                and now - last_progress_at >= progress_interval_s
            ):
                elapsed_s = now - started_at
                print(
                    "Fixed-window integration: "
                    f"file {file_index}/{len(grouped_events)}, "
                    f"{integrated_events:,}/{len(event_table):,} retained events, "
                    f"{scanned_events:,} raw events scanned "
                    f"({elapsed_s:.0f} s elapsed)"
                )
                last_progress_at = now

        missing_segments = set(request_order_by_segment).difference(found_segments)
        if missing_segments:
            missing_preview = sorted(missing_segments)[:10]
            raise ValueError(
                f"Could not reread {len(missing_segments)} requested segments from "
                f"{event_file}; first missing values: {missing_preview}"
            )

    elapsed_s = time.monotonic() - started_at
    print(
        f"Fixed-window integration complete: {integrated_events:,} retained "
        f"events from {len(grouped_events)} files in {elapsed_s:.1f} s"
    )

    return time_ns_ref, charges_mV_ns


def stream_event_qdc_diagnostics(
    events,
    *,
    population_column="qdc_population",
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    peak_snr_threshold=8.0,
    peak_prominence_snr=6.0,
    peak_distance_samples=None,
    peak_width_samples=None,
    pedestal_qdc_window_ns=(0.0, 80.0),
    pedestal_full_waveform_baseline_sidebands_ns=((0.0, 30.0), (70.0, 100.0)),
    pedestal_waveforms_to_keep=10,
    peak_waveforms_to_keep=10,
    multi_peak_waveforms_to_keep=10,
    oscillatory_waveforms_to_keep=10,
    generic_bad_shape_waveforms_to_keep=10,
    small_additional_peak_waveforms_to_keep=10,
    small_additional_peak_blue_waveforms_to_keep=10,
    signal_template_reference_snr=15.0,
    signal_template_reference_waveforms_to_keep=200,
    signal_shape_template_relative_time_ns=None,
    signal_shape_template_normalized=None,
    shape_pedestal_time_blocks=10,
    shape_pedestal_edge_blocks=2,
    shape_pedestal_local_rms_window_ns=4.0,
    shape_pedestal_rms_variation_max=np.inf,
    shape_worst_waveforms_to_keep=10,
    shape_signal_reject_upper_fraction=0.05,
    shape_pedestal_growth_ratio_max=np.inf,
    shape_boundary_waveforms_to_keep=10,
    pedestal_diagnostic_qdc_range=(4.0, 6.0),
    pedestal_diagnostic_max_waveforms=8,
    peak_diagnostic_qdc_range=None,
    peak_diagnostic_max_waveforms=10,
    wide_peak_diagnostic_min_width_ns=None,
    wide_peak_diagnostic_max_waveforms=10,
    read_chunk_size=128,
    progress_interval_s=10.0,
):
    """Stream selected pedestal/peak events and retain only reduced results.

    Pedestal events are baseline-subtracted with their full-trace mean and
    integrated over ``pedestal_qdc_window_ns``. A second pedestal charge spans
    the complete trace after subtracting a baseline learned from
    ``pedestal_full_waveform_baseline_sidebands_ns``; using the full-trace mean
    for that integral would force it to zero by definition.
    Peak events use the configured baseline window and are integrated from
    strongest-candidate time minus one FWHM through candidate time plus two
    FWHM. For comparison, the raw full trace is also integrated without any
    baseline subtraction for every requested pedestal and peak event. Full
    waveforms are retained only for explicitly requested samples. When a
    normalized signal-shape template is supplied, signal template mismatch
    and pedestal noise-growth metrics are calculated in this same stream.
    """
    from scipy.signal import find_peaks, peak_prominences, peak_widths

    required = {"event_file", "event_segment", population_column}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    populations = events[population_column].astype(str)
    unknown = set(populations).difference(
        {
            "pedestal", "peak", "multi_peak", "oscillatory",
            "generic_bad_shape", "small_additional_peak",
            "small_additional_peak_blue",
        }
    )
    if unknown:
        raise ValueError(f"Unknown QDC populations: {sorted(unknown)}")
    read_chunk_size = int(read_chunk_size)
    if read_chunk_size < 1:
        raise ValueError("read_chunk_size must be at least 1")

    shape_template_supplied = (
        signal_shape_template_relative_time_ns is not None
        or signal_shape_template_normalized is not None
    )
    if shape_template_supplied and (
        signal_shape_template_relative_time_ns is None
        or signal_shape_template_normalized is None
    ):
        raise ValueError("Both signal shape-template arrays must be supplied")
    if shape_template_supplied:
        shape_template_time = np.asarray(
            signal_shape_template_relative_time_ns, dtype=float
        )
        shape_template = np.asarray(signal_shape_template_normalized, dtype=float)
        if (
            shape_template_time.ndim != 1
            or shape_template.shape != shape_template_time.shape
            or len(shape_template_time) < 2
            or not np.all(np.diff(shape_template_time) > 0)
            or not np.isfinite(shape_template).all()
        ):
            raise ValueError("Signal shape-template arrays must be finite matching 1D arrays")
        shape_template_norm = float(np.sqrt(np.mean(shape_template**2)))
        if shape_template_norm <= 0.0:
            raise ValueError("Signal shape template has zero norm")
        shape_pedestal_time_blocks = int(shape_pedestal_time_blocks)
        shape_pedestal_edge_blocks = int(shape_pedestal_edge_blocks)
        shape_pedestal_local_rms_window_ns = float(
            shape_pedestal_local_rms_window_ns
        )
        shape_pedestal_rms_variation_max = float(
            shape_pedestal_rms_variation_max
        )
        shape_worst_waveforms_to_keep = int(shape_worst_waveforms_to_keep)
        shape_signal_reject_upper_fraction = float(
            shape_signal_reject_upper_fraction
        )
        shape_pedestal_growth_ratio_max = float(
            shape_pedestal_growth_ratio_max
        )
        shape_boundary_waveforms_to_keep = int(
            shape_boundary_waveforms_to_keep
        )
        if shape_pedestal_time_blocks < 2:
            raise ValueError("shape_pedestal_time_blocks must be at least 2")
        if not 1 <= shape_pedestal_edge_blocks <= shape_pedestal_time_blocks // 2:
            raise ValueError("shape_pedestal_edge_blocks must fit at both trace ends")
        if shape_pedestal_local_rms_window_ns <= 0.0:
            raise ValueError("shape_pedestal_local_rms_window_ns must be positive")
        if shape_pedestal_rms_variation_max < 0.0:
            raise ValueError("shape_pedestal_rms_variation_max must be non-negative")
        if shape_worst_waveforms_to_keep < 0:
            raise ValueError("shape_worst_waveforms_to_keep must be non-negative")
        if not 0.0 <= shape_signal_reject_upper_fraction < 1.0:
            raise ValueError("shape_signal_reject_upper_fraction must be in [0, 1)")
        if shape_pedestal_growth_ratio_max <= 0.0:
            raise ValueError("shape_pedestal_growth_ratio_max must be positive")
        if shape_boundary_waveforms_to_keep < 0:
            raise ValueError("shape_boundary_waveforms_to_keep must be non-negative")

    table_columns = ["event_file", "event_segment", population_column]
    if shape_template_supplied:
        if "peak_time_ns" not in events.columns:
            raise ValueError("Shape-quality calculation requires peak_time_ns")
        table_columns.append("peak_time_ns")
    table = events.loc[:, table_columns].copy()
    table["_request_order"] = np.arange(len(table))
    table["_population_order"] = table.groupby(population_column).cumcount()
    qdc_values = np.full(len(table), np.nan)
    full_waveform_qdc_values = np.full(len(table), np.nan)
    raw_full_waveform_qdc_values = np.full(len(table), np.nan)
    max_amplitude_values = np.full(len(table), np.nan)
    signal_template_mismatch = np.full(len(table), np.nan)
    pedestal_noise_growth_ratio = np.full(len(table), np.nan)
    pedestal_local_rms_relative_span = np.full(len(table), np.nan)
    worst_signal_shape_records = []
    worst_pedestal_growth_records = []
    worst_pedestal_rms_variation_records = []
    signal_shape_boundary_candidates = []
    accepted_pedestal_growth_records = []
    accepted_pedestal_rms_variation_records = []
    n_shape_signals = int(np.sum(populations.to_numpy() == "peak"))
    signal_shape_boundary_candidate_limit = min(
        n_shape_signals,
        int(np.ceil(shape_signal_reject_upper_fraction * n_shape_signals))
        + shape_boundary_waveforms_to_keep,
    )
    plot_records = {
        "pedestal": [], "peak": [], "multi_peak": [], "oscillatory": [],
        "generic_bad_shape": [],
        "small_additional_peak": [],
        "small_additional_peak_blue": [],
    }
    diagnostic_records = []
    peak_diagnostic_records = []
    wide_peak_diagnostic_records = []
    signal_template_reference_records = []
    time_ns_ref = None
    started_at = time.monotonic()
    last_progress_at = started_at
    processed = 0

    def integrate_window(time_ns, waveform, start_ns, stop_ns):
        if start_ns < time_ns[0] or stop_ns > time_ns[-1] or start_ns >= stop_ns:
            return np.nan
        inner = (time_ns > start_ns) & (time_ns < stop_ns)
        integration_time = np.concatenate(([start_ns], time_ns[inner], [stop_ns]))
        integration_voltage = np.interp(integration_time, time_ns, waveform)
        return float(np.trapezoid(integration_voltage, integration_time))

    def robust_rms(values):
        values = np.asarray(values, dtype=float)
        center = float(np.median(values))
        estimate = 1.4826 * float(np.median(np.abs(values - center)))
        return estimate if estimate > 0.0 else float(np.std(values))

    def local_robust_rms(time_ns, values, window_ns):
        values = np.asarray(values, dtype=float)
        was_one_dimensional = values.ndim == 1
        if was_one_dimensional:
            values = values[None, :]
        elif values.ndim != 2:
            raise ValueError("Local-RMS input must be one- or two-dimensional")
        dt_ns = float(np.median(np.diff(time_ns)))
        samples_per_window = int(round(window_ns / dt_ns))
        if samples_per_window < 2:
            raise ValueError(
                "Pedestal local-RMS window contains fewer than two samples"
            )
        n_windows = values.shape[1] // samples_per_window
        if n_windows < 3:
            raise ValueError(
                "Pedestal local-RMS calculation requires at least three windows"
            )
        blocks = values[:, :n_windows * samples_per_window].reshape(
            len(values), n_windows, samples_per_window
        )
        centers = np.median(blocks, axis=2)
        estimates = 1.4826 * np.median(
            np.abs(blocks - centers[:, :, None]), axis=2
        )
        zero = estimates <= 0.0
        if np.any(zero):
            standard_deviations = np.std(blocks, axis=2)
            estimates[zero] = standard_deviations[zero]
        return estimates[0] if was_one_dimensional else estimates

    def retain_top_shape(records, record, limit):
        if not shape_template_supplied or limit == 0:
            return
        records.append(record)
        records.sort(key=lambda item: item["metric"], reverse=True)
        del records[limit:]

    def subtract_sideband_baseline(time_ns, raw_waveform):
        sideband_mask = np.zeros(len(time_ns), dtype=bool)
        for sideband in pedestal_full_waveform_baseline_sidebands_ns:
            if len(sideband) != 2:
                raise ValueError("Each pedestal baseline sideband must have two bounds")
            start_ns, stop_ns = map(float, sideband)
            if start_ns >= stop_ns:
                raise ValueError("Pedestal baseline sideband start must be below stop")
            sideband_mask |= (time_ns >= start_ns) & (time_ns <= stop_ns)
        if not np.any(sideband_mask):
            raise ValueError(
                "Pedestal full-waveform baseline sidebands contain no samples"
            )
        waveform = np.asarray(raw_waveform, dtype=float)
        if baseline_reference_mV is not None:
            waveform = waveform - np.asarray(baseline_reference_mV, dtype=float)
        sideband_mean_mV = float(np.mean(waveform[sideband_mask]))
        return waveform - sideband_mean_mV, sideband_mean_mV

    def candidate_metrics(time_ns, positive_waveform, baseline_rms):
        candidates, _ = find_peaks(positive_waveform)
        if not len(candidates):
            return None
        candidate_idx = int(candidates[np.argmax(positive_waveform[candidates])])
        prominence = float(peak_prominences(positive_waveform, [candidate_idx])[0][0])
        widths, width_heights, left_ips, right_ips = peak_widths(
            positive_waveform, [candidate_idx], rel_height=0.5
        )
        dt_ns = float(np.mean(np.diff(time_ns)))
        width_samples = float(widths[0])
        failed = []
        amplitude = float(positive_waveform[candidate_idx])
        if amplitude < float(peak_snr_threshold) * baseline_rms:
            failed.append("height/SNR")
        if peak_prominence_snr is not None and prominence < float(
            peak_prominence_snr
        ) * baseline_rms:
            failed.append("prominence")
        if peak_width_samples is not None:
            width_min, width_max = peak_width_samples
            if width_min is not None and width_samples < float(width_min):
                failed.append("width below minimum")
            if width_max is not None and width_samples > float(width_max):
                failed.append("width above maximum")
        if not failed:
            failed.append("distance interaction with another candidate")
        return {
            "candidate_idx": candidate_idx,
            "time_ns": float(time_ns[candidate_idx]),
            "amplitude_mV": amplitude,
            "snr": amplitude / baseline_rms if baseline_rms > 0 else np.inf,
            "prominence_snr": prominence / baseline_rms if baseline_rms > 0 else np.inf,
            "width_samples": width_samples,
            "width_ns": width_samples * dt_ns,
            "width_level_mV": -float(width_heights[0]),
            "width_left_ns": float(np.interp(left_ips[0], np.arange(len(time_ns)), time_ns)),
            "width_right_ns": float(np.interp(right_ips[0], np.arange(len(time_ns)), time_ns)),
            "failed_checks": failed,
        }

    def qualifying_peak_metrics(
        time_ns, positive_waveform, baseline_rms, *, enforce_width=True
    ):
        minimum_height = float(peak_snr_threshold) * baseline_rms
        prominence = (
            None
            if peak_prominence_snr is None
            else float(peak_prominence_snr) * baseline_rms
        )
        candidates, properties = find_peaks(
            positive_waveform,
            height=minimum_height,
            prominence=prominence,
            distance=peak_distance_samples,
            width=(peak_width_samples if enforce_width else None),
        )
        if not len(candidates):
            return []
        prominences = properties.get("prominences")
        if prominences is None:
            prominences = peak_prominences(positive_waveform, candidates)[0]
        widths = properties.get("widths")
        width_heights = properties.get("width_heights")
        left_ips = properties.get("left_ips")
        right_ips = properties.get("right_ips")
        if any(value is None for value in (widths, width_heights, left_ips, right_ips)):
            widths, width_heights, left_ips, right_ips = peak_widths(
                positive_waveform, candidates, rel_height=0.5
            )
        dt_ns = float(np.mean(np.diff(time_ns)))
        return [
            {
                "candidate_idx": int(candidate),
                "time_ns": float(time_ns[candidate]),
                "amplitude_mV": float(positive_waveform[candidate]),
                "snr": (
                    float(positive_waveform[candidate]) / baseline_rms
                    if baseline_rms > 0 else np.inf
                ),
                "prominence_snr": (
                    float(prominences[index]) / baseline_rms
                    if baseline_rms > 0 else np.inf
                ),
                "width_samples": float(widths[index]),
                "width_ns": float(widths[index]) * dt_ns,
                "width_level_mV": -float(width_heights[index]),
                "width_left_ns": float(
                    np.interp(left_ips[index], np.arange(len(time_ns)), time_ns)
                ),
                "width_right_ns": float(
                    np.interp(right_ips[index], np.arange(len(time_ns)), time_ns)
                ),
            }
            for index, candidate in enumerate(candidates)
        ]

    for event_file, group in table.groupby("event_file", sort=False):
        segment_rows = {
            int(row["event_segment"]): row.to_dict()
            for _, row in group.iterrows()
        }
        found = set()
        for chunk in iter_keysight_chunks(
            [event_file], channel=channel, chunk_size=read_chunk_size
        ):
            time_ns = np.asarray(chunk["time_ns"], dtype=float)
            time_ns = time_ns - time_ns[0]
            if time_ns_ref is None:
                time_ns_ref = time_ns.copy()
            elif not np.allclose(time_ns_ref, time_ns, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Time axis changed in {event_file}")
            chunk_segments = np.asarray(chunk["segment_numbers"], dtype=int)
            selected = [
                (position, segment_rows[int(segment)])
                for position, segment in enumerate(chunk_segments)
                if int(segment) in segment_rows
            ]
            if not selected:
                continue
            positions = np.asarray([item[0] for item in selected], dtype=int)
            raw_waveforms = np.asarray(chunk["voltage_mV"])[positions]
            local_rms_by_selected_index = {}
            if shape_template_supplied:
                pedestal_selected_indices = [
                    index
                    for index, (_, row) in enumerate(selected)
                    if str(row[population_column]) == "pedestal"
                ]
                if pedestal_selected_indices:
                    pedestal_local_rms = local_robust_rms(
                        time_ns,
                        raw_waveforms[pedestal_selected_indices],
                        shape_pedestal_local_rms_window_ns,
                    )
                    local_rms_by_selected_index = dict(zip(
                        pedestal_selected_indices, pedestal_local_rms
                    ))
            for selected_index, (raw_waveform, (_, row)) in enumerate(
                zip(raw_waveforms, selected)
            ):
                population = str(row[population_column])
                request_order = int(row["_request_order"])
                population_order = int(row["_population_order"])
                raw_full_waveform_qdc_values[request_order] = float(
                    np.trapezoid(-np.asarray(raw_waveform, dtype=float), time_ns)
                )
                if population == "pedestal":
                    full_charge_waveform, full_charge_baseline_mean_mV = (
                        subtract_sideband_baseline(time_ns, raw_waveform)
                    )
                    full_waveform_qdc_values[request_order] = float(
                        np.trapezoid(-full_charge_waveform, time_ns)
                    )
                    if baseline_reference_mV is not None:
                        waveform = raw_waveform - np.asarray(baseline_reference_mV)
                    else:
                        waveform = np.asarray(raw_waveform, dtype=float)
                    waveform = waveform - np.mean(waveform)
                    baseline_rms = float(np.std(waveform))
                    positive = -waveform
                    metrics = candidate_metrics(time_ns, positive, baseline_rms)
                    qualifying_peaks = qualifying_peak_metrics(
                        time_ns, positive, baseline_rms
                    )
                    signal_like_peaks = qualifying_peak_metrics(
                        time_ns, positive, baseline_rms, enforce_width=False
                    )
                    accepted_peak_metrics = (
                        max(qualifying_peaks, key=lambda peak: peak["amplitude_mV"])
                        if qualifying_peaks else None
                    )
                    qdc = integrate_window(
                        time_ns, positive, *pedestal_qdc_window_ns
                    )
                    if shape_template_supplied:
                        block_values = np.array([
                            robust_rms(block)
                            for block in np.array_split(
                                raw_waveform, shape_pedestal_time_blocks
                            )
                        ])
                        early_rms = float(np.median(
                            block_values[:shape_pedestal_edge_blocks]
                        ))
                        late_rms = float(np.median(
                            block_values[-shape_pedestal_edge_blocks:]
                        ))
                        growth_metric = (
                            late_rms / early_rms if early_rms > 0.0 else np.inf
                        )
                        pedestal_noise_growth_ratio[request_order] = growth_metric
                        local_rms_values = local_rms_by_selected_index[
                            selected_index
                        ]
                        local_rms_median = float(np.median(local_rms_values))
                        local_rms_low, local_rms_high = np.quantile(
                            local_rms_values, (0.10, 0.90)
                        )
                        if local_rms_median > 0.0:
                            rms_variation_metric = float(
                                (local_rms_high - local_rms_low)
                                / local_rms_median
                            )
                        elif local_rms_high == local_rms_low:
                            rms_variation_metric = 0.0
                        else:
                            rms_variation_metric = np.inf
                        pedestal_local_rms_relative_span[
                            request_order
                        ] = rms_variation_metric
                        growth_record = {
                            "metric": growth_metric,
                            "qdc_mV_ns": qdc,
                            "event_file": str(event_file),
                            "event_segment": int(row["event_segment"]),
                            "waveform_mV": raw_waveform - np.median(raw_waveform),
                            "block_rms_mV": block_values,
                        }
                        retain_top_shape(
                            worst_pedestal_growth_records,
                            growth_record,
                            shape_worst_waveforms_to_keep,
                        )
                        if growth_metric <= shape_pedestal_growth_ratio_max:
                            retain_top_shape(
                                accepted_pedestal_growth_records,
                                growth_record,
                                shape_boundary_waveforms_to_keep,
                            )
                        rms_variation_record = {
                            "metric": rms_variation_metric,
                            "qdc_mV_ns": qdc,
                            "event_file": str(event_file),
                            "event_segment": int(row["event_segment"]),
                            "waveform_mV": raw_waveform - np.median(raw_waveform),
                            "local_rms_mV": local_rms_values,
                        }
                        retain_top_shape(
                            worst_pedestal_rms_variation_records,
                            rms_variation_record,
                            shape_worst_waveforms_to_keep,
                        )
                        if (
                            rms_variation_metric
                            <= shape_pedestal_rms_variation_max
                        ):
                            retain_top_shape(
                                accepted_pedestal_rms_variation_records,
                                rms_variation_record,
                                shape_boundary_waveforms_to_keep,
                            )
                    keep_plot = population_order < int(pedestal_waveforms_to_keep)
                    keep_diagnostic = (
                        len(diagnostic_records) < int(pedestal_diagnostic_max_waveforms)
                        and float(pedestal_diagnostic_qdc_range[0]) <= qdc
                        <= float(pedestal_diagnostic_qdc_range[1])
                    )
                else:
                    waveform, info = baseline_subtraction(
                        time_ns,
                        np.asarray(raw_waveform)[None, :],
                        baseline_window_ns=baseline_window_ns,
                        baseline_reference_time_ns=baseline_reference_time_ns,
                        baseline_reference_mV=baseline_reference_mV,
                    )
                    waveform = waveform[0]
                    baseline_rms = float(info["baseline_rms_mV"][0])
                    positive = -waveform
                    metrics = candidate_metrics(time_ns, positive, baseline_rms)
                    qualifying_peaks = qualifying_peak_metrics(
                        time_ns, positive, baseline_rms
                    )
                    signal_like_peaks = qualifying_peak_metrics(
                        time_ns, positive, baseline_rms, enforce_width=False
                    )
                    accepted_peak_metrics = (
                        max(qualifying_peaks, key=lambda peak: peak["amplitude_mV"])
                        if qualifying_peaks else None
                    )
                    if accepted_peak_metrics is None:
                        qdc = np.nan
                    else:
                        qdc = integrate_window(
                            time_ns,
                            positive,
                            accepted_peak_metrics["time_ns"]
                            - accepted_peak_metrics["width_ns"],
                            accepted_peak_metrics["time_ns"]
                            + 2.0 * accepted_peak_metrics["width_ns"],
                        )
                    if shape_template_supplied and population == "peak":
                        shape_peak_time = float(row["peak_time_ns"])
                        shape_sample_times = shape_peak_time + shape_template_time
                        if (
                            shape_sample_times[0] >= time_ns[0]
                            and shape_sample_times[-1] <= time_ns[-1]
                        ):
                            aligned_shape = np.interp(
                                shape_sample_times, time_ns, positive
                            )
                            shape_amplitude = float(np.max(aligned_shape))
                            if shape_amplitude > 0.0:
                                normalized_shape = aligned_shape / shape_amplitude
                                mismatch_metric = float(
                                    np.sqrt(np.mean(
                                        (normalized_shape - shape_template) ** 2
                                    ))
                                    / shape_template_norm
                                )
                                signal_template_mismatch[
                                    request_order
                                ] = mismatch_metric
                                mismatch_record = {
                                    "metric": mismatch_metric,
                                    "qdc_mV_ns": qdc,
                                    "event_file": str(event_file),
                                    "event_segment": int(row["event_segment"]),
                                    "waveform_mV": waveform.copy(),
                                    "peak_time_ns": shape_peak_time,
                                    "normalized_shape": normalized_shape,
                                }
                                retain_top_shape(
                                    worst_signal_shape_records,
                                    mismatch_record,
                                    shape_worst_waveforms_to_keep,
                                )
                                retain_top_shape(
                                    signal_shape_boundary_candidates,
                                    mismatch_record,
                                    signal_shape_boundary_candidate_limit,
                                )
                    plot_limit = {
                        "peak": peak_waveforms_to_keep,
                        "multi_peak": multi_peak_waveforms_to_keep,
                        "oscillatory": oscillatory_waveforms_to_keep,
                        "generic_bad_shape": generic_bad_shape_waveforms_to_keep,
                        "small_additional_peak": (
                            small_additional_peak_waveforms_to_keep
                        ),
                        "small_additional_peak_blue": (
                            small_additional_peak_blue_waveforms_to_keep
                        ),
                    }[population]
                    keep_plot = population_order < int(plot_limit)
                    keep_diagnostic = False
                    keep_peak_diagnostic = (
                        population == "peak"
                        and peak_diagnostic_qdc_range is not None
                        and len(peak_diagnostic_records)
                        < int(peak_diagnostic_max_waveforms)
                        and float(peak_diagnostic_qdc_range[0]) <= qdc
                        <= float(peak_diagnostic_qdc_range[1])
                    )
                    keep_wide_peak_diagnostic = (
                        population == "peak"
                        and wide_peak_diagnostic_min_width_ns is not None
                        and accepted_peak_metrics is not None
                        and accepted_peak_metrics["width_ns"]
                        > float(wide_peak_diagnostic_min_width_ns)
                        and len(wide_peak_diagnostic_records)
                        < int(wide_peak_diagnostic_max_waveforms)
                    )

                qdc_values[request_order] = qdc
                max_amplitude_values[request_order] = float(np.max(positive))
                record = {
                    "request_order": request_order,
                    "population_order": population_order,
                    "event_file": str(event_file),
                    "event_segment": int(row["event_segment"]),
                    "baseline_rms_mV": baseline_rms,
                    "qdc_mV_ns": qdc,
                    "full_waveform_qdc_mV_ns": full_waveform_qdc_values[
                        request_order
                    ],
                    "full_waveform_sideband_baseline_mean_mV": (
                        full_charge_baseline_mean_mV
                        if population == "pedestal" else np.nan
                    ),
                    "metrics": metrics,
                    "accepted_peak_metrics": accepted_peak_metrics,
                    "qualifying_peaks": qualifying_peaks,
                    "signal_like_peaks": signal_like_peaks,
                }
                if keep_plot:
                    record["waveform_mV"] = np.asarray(waveform).copy()
                    plot_records[population].append(record)
                if (
                    population == "peak"
                    and accepted_peak_metrics is not None
                    and accepted_peak_metrics["snr"]
                    >= float(signal_template_reference_snr)
                    and len(signal_template_reference_records)
                    < int(signal_template_reference_waveforms_to_keep)
                ):
                    template_record = dict(record)
                    template_record["waveform_mV"] = np.asarray(waveform).copy()
                    signal_template_reference_records.append(template_record)
                if keep_diagnostic:
                    diagnostic_record = dict(record)
                    diagnostic_record["waveform_mV"] = np.asarray(waveform).copy()
                    diagnostic_records.append(diagnostic_record)
                if population != "pedestal" and keep_peak_diagnostic:
                    peak_diagnostic_record = dict(record)
                    peak_diagnostic_record["waveform_mV"] = np.asarray(waveform).copy()
                    peak_diagnostic_records.append(peak_diagnostic_record)
                if population != "pedestal" and keep_wide_peak_diagnostic:
                    wide_peak_record = dict(record)
                    wide_peak_record["waveform_mV"] = np.asarray(waveform).copy()
                    wide_peak_diagnostic_records.append(wide_peak_record)
                found.add(int(row["event_segment"]))
                processed += 1

            now = time.monotonic()
            if progress_interval_s is not None and now - last_progress_at >= float(
                progress_interval_s
            ):
                print(
                    f"Streaming QDC: {processed:,}/{len(table):,} selected events "
                    f"processed ({now - started_at:.0f} s elapsed)"
                )
                last_progress_at = now
        missing_segments = set(segment_rows).difference(found)
        if missing_segments:
            raise ValueError(
                f"Could not reread {len(missing_segments)} selected segments from {event_file}"
            )

    print(
        f"Streaming QDC complete: {processed:,} events in "
        f"{time.monotonic() - started_at:.1f} s; retained "
        f"{sum(map(len, plot_records.values()))} plot waveforms and "
        f"{len(diagnostic_records)} diagnostic waveforms in RAM"
    )
    population_array = populations.to_numpy()
    return {
        "time_ns": time_ns_ref,
        "pedestal_qdc_mV_ns": qdc_values[population_array == "pedestal"],
        "pedestal_full_waveform_qdc_mV_ns": full_waveform_qdc_values[
            population_array == "pedestal"
        ],
        "peak_qdc_mV_ns": qdc_values[population_array == "peak"],
        "pedestal_raw_full_waveform_qdc_mV_ns": raw_full_waveform_qdc_values[
            population_array == "pedestal"
        ],
        "peak_raw_full_waveform_qdc_mV_ns": raw_full_waveform_qdc_values[
            population_array == "peak"
        ],
        "pedestal_max_amplitude_mV": max_amplitude_values[
            population_array == "pedestal"
        ],
        "peak_max_amplitude_mV": max_amplitude_values[population_array == "peak"],
        "signal_template_mismatch": signal_template_mismatch[
            population_array == "peak"
        ],
        "pedestal_noise_growth_ratio": pedestal_noise_growth_ratio[
            population_array == "pedestal"
        ],
        "pedestal_local_rms_relative_span": pedestal_local_rms_relative_span[
            population_array == "pedestal"
        ],
        "worst_signal_shape_records": worst_signal_shape_records,
        "worst_pedestal_growth_records": worst_pedestal_growth_records,
        "worst_pedestal_rms_variation_records": (
            worst_pedestal_rms_variation_records
        ),
        "signal_shape_boundary_candidates": signal_shape_boundary_candidates,
        "accepted_pedestal_growth_records": accepted_pedestal_growth_records,
        "accepted_pedestal_rms_variation_records": (
            accepted_pedestal_rms_variation_records
        ),
        "multi_peak_qdc_mV_ns": qdc_values[population_array == "multi_peak"],
        "multi_peak_max_amplitude_mV": max_amplitude_values[
            population_array == "multi_peak"
        ],
        "oscillatory_qdc_mV_ns": qdc_values[
            population_array == "oscillatory"
        ],
        "generic_bad_shape_qdc_mV_ns": qdc_values[
            population_array == "generic_bad_shape"
        ],
        "small_additional_peak_qdc_mV_ns": qdc_values[
            population_array == "small_additional_peak"
        ],
        "plot_records": plot_records,
        "pedestal_diagnostic_records": diagnostic_records,
        "peak_diagnostic_records": peak_diagnostic_records,
        "wide_peak_diagnostic_records": wide_peak_diagnostic_records,
        "signal_template_reference_records": signal_template_reference_records,
    }


def stream_event_charge_method_comparison(
    events,
    *,
    population_column="qdc_population",
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    pedestal_qdc_window_ns=(0.0, 80.0),
    read_chunk_size=128,
    progress_interval_s=10.0,
):
    """Stream event-specific and raw full-trace charges without peak finding."""
    required = {
        "event_file", "event_segment", population_column,
        "peak_time_ns", "peak_width_ns",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing charge-comparison columns: {sorted(missing)}")
    populations = events[population_column].astype(str)
    unknown = set(populations).difference({"pedestal", "peak"})
    if unknown:
        raise ValueError(f"Unknown charge populations: {sorted(unknown)}")
    read_chunk_size = int(read_chunk_size)
    if read_chunk_size < 1:
        raise ValueError("read_chunk_size must be at least 1")

    table = events.loc[:, [
        "event_file", "event_segment", population_column,
        "peak_time_ns", "peak_width_ns",
    ]].copy()
    table["_request_order"] = np.arange(len(table))
    event_specific = np.full(len(table), np.nan)
    raw_full_trace = np.full(len(table), np.nan)
    time_ns_ref = None
    processed = 0
    started_at = time.monotonic()
    last_progress_at = started_at

    def integrate_window(time_ns, waveform, start_ns, stop_ns):
        if start_ns < time_ns[0] or stop_ns > time_ns[-1] or start_ns >= stop_ns:
            return np.nan
        inner = (time_ns > start_ns) & (time_ns < stop_ns)
        integration_time = np.concatenate(([start_ns], time_ns[inner], [stop_ns]))
        integration_voltage = np.interp(integration_time, time_ns, waveform)
        return float(np.trapezoid(integration_voltage, integration_time))

    for event_file, group in table.groupby("event_file", sort=False):
        segment_rows = {
            int(row["event_segment"]): row for _, row in group.iterrows()
        }
        found = set()
        for chunk in iter_keysight_chunks(
            [event_file], channel=channel, chunk_size=read_chunk_size
        ):
            time_ns = np.asarray(chunk["time_ns"], dtype=float)
            time_ns = time_ns - time_ns[0]
            if time_ns_ref is None:
                time_ns_ref = time_ns.copy()
            elif not np.allclose(time_ns_ref, time_ns, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Time axis changed in {event_file}")
            chunk_segments = np.asarray(chunk["segment_numbers"], dtype=int)
            selected = [
                (position, segment_rows[int(segment)])
                for position, segment in enumerate(chunk_segments)
                if int(segment) in segment_rows
            ]
            if not selected:
                continue
            positions = np.asarray([item[0] for item in selected], dtype=int)
            raw_waveforms = np.asarray(chunk["voltage_mV"])[positions]
            for raw_waveform, (_, row) in zip(raw_waveforms, selected):
                request_order = int(row["_request_order"])
                population = str(row[population_column])
                raw_waveform = np.asarray(raw_waveform, dtype=float)
                raw_full_trace[request_order] = float(
                    np.trapezoid(-raw_waveform, time_ns)
                )
                if population == "pedestal":
                    if baseline_reference_mV is not None:
                        waveform = raw_waveform - np.asarray(
                            baseline_reference_mV, dtype=float
                        )
                    else:
                        waveform = raw_waveform.copy()
                    positive = -(waveform - np.mean(waveform))
                    event_specific[request_order] = integrate_window(
                        time_ns, positive, *pedestal_qdc_window_ns
                    )
                else:
                    waveform, _ = baseline_subtraction(
                        time_ns, raw_waveform[None, :],
                        baseline_window_ns=baseline_window_ns,
                        baseline_reference_time_ns=baseline_reference_time_ns,
                        baseline_reference_mV=baseline_reference_mV,
                    )
                    peak_time = float(row["peak_time_ns"])
                    peak_width = float(row["peak_width_ns"])
                    if np.isfinite(peak_time) and np.isfinite(peak_width):
                        event_specific[request_order] = integrate_window(
                            time_ns, -waveform[0],
                            peak_time - peak_width,
                            peak_time + 2.0 * peak_width,
                        )
                found.add(int(row["event_segment"]))
                processed += 1

            now = time.monotonic()
            if progress_interval_s is not None and now - last_progress_at >= float(
                progress_interval_s
            ):
                print(
                    f"Streaming voltage QDC methods: {processed:,}/{len(table):,} "
                    f"events ({now - started_at:.0f} s elapsed)"
                )
                last_progress_at = now
        missing_segments = set(segment_rows).difference(found)
        if missing_segments:
            raise ValueError(
                f"Could not reread {len(missing_segments)} charge-comparison "
                f"segments from {event_file}"
            )

    population_array = populations.to_numpy()
    print(
        f"Streaming voltage QDC methods complete: {processed:,} events in "
        f"{time.monotonic() - started_at:.1f} s"
    )
    return {
        "pedestal_event_specific_qdc_mV_ns": event_specific[
            population_array == "pedestal"
        ],
        "peak_event_specific_qdc_mV_ns": event_specific[
            population_array == "peak"
        ],
        "pedestal_raw_full_waveform_qdc_mV_ns": raw_full_trace[
            population_array == "pedestal"
        ],
        "peak_raw_full_waveform_qdc_mV_ns": raw_full_trace[
            population_array == "peak"
        ],
    }


def stream_shape_quality_diagnostics(
    events,
    *,
    signal_template_relative_time_ns,
    signal_template_normalized,
    population_column="shape_population",
    qdc_column="diagnostic_qdc_mV_ns",
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    pedestal_time_blocks=5,
    pedestal_edge_blocks=2,
    worst_waveforms_to_keep=10,
    read_chunk_size=128,
    progress_interval_s=10.0,
):
    """Stream template-mismatch and pedestal noise-growth diagnostics."""
    required = {
        "event_file", "event_segment", population_column, qdc_column,
        "peak_time_ns",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing shape-diagnostic columns: {sorted(missing)}")
    populations = events[population_column].astype(str)
    unknown = set(populations).difference({"pedestal", "signal"})
    if unknown:
        raise ValueError(f"Unknown shape populations: {sorted(unknown)}")

    template_time = np.asarray(signal_template_relative_time_ns, dtype=float)
    template = np.asarray(signal_template_normalized, dtype=float)
    if template_time.ndim != 1 or template.shape != template_time.shape:
        raise ValueError("Signal template time and voltage must be matching 1D arrays")
    if len(template_time) < 2 or not np.all(np.diff(template_time) > 0):
        raise ValueError("Signal template time must be strictly increasing")
    if not np.isfinite(template).all():
        raise ValueError("Signal template contains non-finite values")
    template_norm = float(np.sqrt(np.mean(template**2)))
    if template_norm <= 0.0:
        raise ValueError("Signal template has zero norm")

    pedestal_time_blocks = int(pedestal_time_blocks)
    pedestal_edge_blocks = int(pedestal_edge_blocks)
    worst_waveforms_to_keep = int(worst_waveforms_to_keep)
    read_chunk_size = int(read_chunk_size)
    if pedestal_time_blocks < 2:
        raise ValueError("pedestal_time_blocks must be at least 2")
    if not 1 <= pedestal_edge_blocks <= pedestal_time_blocks // 2:
        raise ValueError("pedestal_edge_blocks must fit at both ends of the trace")
    if worst_waveforms_to_keep < 0 or read_chunk_size < 1:
        raise ValueError("Diagnostic retention must be non-negative and chunk size positive")

    table = events.loc[:, [
        "event_file", "event_segment", population_column, qdc_column,
        "peak_time_ns",
    ]].copy()
    table["_request_order"] = np.arange(len(table))
    mismatch = np.full(len(table), np.nan)
    growth_ratio = np.full(len(table), np.nan)
    worst_signal_records = []
    worst_pedestal_records = []
    time_ns_ref = None
    processed = 0
    started_at = time.monotonic()
    last_progress_at = started_at

    def robust_rms(values):
        values = np.asarray(values, dtype=float)
        center = float(np.median(values))
        estimate = 1.4826 * float(np.median(np.abs(values - center)))
        return estimate if estimate > 0.0 else float(np.std(values))

    def retain_worst(records, record):
        if worst_waveforms_to_keep == 0:
            return
        records.append(record)
        records.sort(key=lambda item: item["metric"], reverse=True)
        del records[worst_waveforms_to_keep:]

    for event_file, group in table.groupby("event_file", sort=False):
        segment_rows = {
            int(row["event_segment"]): row for _, row in group.iterrows()
        }
        found = set()
        for chunk in iter_keysight_chunks(
            [event_file], channel=channel, chunk_size=read_chunk_size
        ):
            time_ns = np.asarray(chunk["time_ns"], dtype=float)
            time_ns = time_ns - time_ns[0]
            if time_ns_ref is None:
                time_ns_ref = time_ns.copy()
            elif not np.allclose(time_ns_ref, time_ns, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Time axis changed in {event_file}")
            chunk_segments = np.asarray(chunk["segment_numbers"], dtype=int)
            selected = [
                (position, segment_rows[int(segment)])
                for position, segment in enumerate(chunk_segments)
                if int(segment) in segment_rows
            ]
            if not selected:
                continue
            positions = np.asarray([item[0] for item in selected], dtype=int)
            raw_waveforms = np.asarray(chunk["voltage_mV"])[positions]
            for raw_waveform, (_, row) in zip(raw_waveforms, selected):
                request_order = int(row["_request_order"])
                population = str(row[population_column])
                qdc = float(row[qdc_column])
                raw_waveform = np.asarray(raw_waveform, dtype=float)
                if population == "pedestal":
                    block_values = np.array([
                        robust_rms(block)
                        for block in np.array_split(raw_waveform, pedestal_time_blocks)
                    ])
                    early_rms = float(np.median(block_values[:pedestal_edge_blocks]))
                    late_rms = float(np.median(block_values[-pedestal_edge_blocks:]))
                    metric = late_rms / early_rms if early_rms > 0.0 else np.inf
                    growth_ratio[request_order] = metric
                    retain_worst(worst_pedestal_records, {
                        "metric": metric,
                        "qdc_mV_ns": qdc,
                        "event_file": str(event_file),
                        "event_segment": int(row["event_segment"]),
                        "waveform_mV": raw_waveform - np.median(raw_waveform),
                        "block_rms_mV": block_values,
                    })
                else:
                    waveform, _ = baseline_subtraction(
                        time_ns, raw_waveform[None, :],
                        baseline_window_ns=baseline_window_ns,
                        baseline_reference_time_ns=baseline_reference_time_ns,
                        baseline_reference_mV=baseline_reference_mV,
                    )
                    waveform = waveform[0]
                    positive = -waveform
                    peak_time = float(row["peak_time_ns"])
                    sample_times = peak_time + template_time
                    if sample_times[0] < time_ns[0] or sample_times[-1] > time_ns[-1]:
                        found.add(int(row["event_segment"]))
                        processed += 1
                        continue
                    aligned = np.interp(sample_times, time_ns, positive)
                    amplitude = float(np.max(aligned))
                    if amplitude > 0.0:
                        normalized = aligned / amplitude
                        metric = float(
                            np.sqrt(np.mean((normalized - template) ** 2))
                            / template_norm
                        )
                        mismatch[request_order] = metric
                        retain_worst(worst_signal_records, {
                            "metric": metric,
                            "qdc_mV_ns": qdc,
                            "event_file": str(event_file),
                            "event_segment": int(row["event_segment"]),
                            "waveform_mV": waveform.copy(),
                            "peak_time_ns": peak_time,
                            "normalized_shape": normalized,
                        })
                found.add(int(row["event_segment"]))
                processed += 1

            now = time.monotonic()
            if progress_interval_s is not None and now - last_progress_at >= float(
                progress_interval_s
            ):
                print(
                    f"Streaming shape diagnostics: {processed:,}/{len(table):,} "
                    f"events ({now - started_at:.0f} s elapsed)"
                )
                last_progress_at = now
        missing_segments = set(segment_rows).difference(found)
        if missing_segments:
            raise ValueError(
                f"Could not reread {len(missing_segments)} shape-diagnostic "
                f"segments from {event_file}"
            )

    population_array = populations.to_numpy()
    print(
        f"Streaming shape diagnostics complete: {processed:,} events in "
        f"{time.monotonic() - started_at:.1f} s"
    )
    return {
        "time_ns": time_ns_ref,
        "signal_template_mismatch": mismatch[population_array == "signal"],
        "pedestal_noise_growth_ratio": growth_ratio[
            population_array == "pedestal"
        ],
        "worst_signal_records": worst_signal_records,
        "worst_pedestal_records": worst_pedestal_records,
    }


def resolve_cached_event_files(source_files, *, search_roots=(Path("PMT_Data"),)):
    """Resolve raw files recorded in a cache after data directories move.

    Existing cached paths are retained. For a missing path, its basename is
    searched below ``search_roots``. Basenames must resolve uniquely so an old
    cache can never silently select the wrong acquisition file.
    """

    search_roots = tuple(Path(root) for root in search_roots)
    resolved = {}
    remapped = []
    for source_file in source_files:
        cached_path = Path(source_file)
        basename = cached_path.name
        if cached_path.is_file():
            resolved_path = cached_path.resolve()
        else:
            matches = sorted(
                {
                    candidate.resolve()
                    for root in search_roots
                    if root.is_dir()
                    for candidate in root.rglob(basename)
                    if candidate.is_file()
                }
            )
            if not matches:
                raise FileNotFoundError(
                    f"Cached raw file no longer exists: {cached_path}. "
                    f"No file named {basename!r} was found below "
                    f"{[str(root) for root in search_roots]}."
                )
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Cached raw file path is stale and basename {basename!r} "
                    f"is ambiguous; matches: {[str(path) for path in matches]}"
                )
            resolved_path = matches[0]
            remapped.append((cached_path, resolved_path))
        previous = resolved.get(basename)
        if previous is not None and previous != resolved_path:
            raise ValueError(
                f"Cache contains duplicate basename {basename!r} for both "
                f"{previous} and {resolved_path}"
            )
        resolved[basename] = resolved_path

    if remapped:
        old_parents = sorted({str(old.parent) for old, _ in remapped})
        new_parents = sorted({str(new.parent) for _, new in remapped})
        print(
            f"Remapped {len(remapped)} stale cached raw-file paths by basename: "
            f"{old_parents} -> {new_parents}"
        )
    return resolved


def load_event_waveforms(
    events,
    *,
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    time_origin="zero",
    subtract_baseline=True,
):
    """Load specific events, optionally applying the configured baseline subtraction.

    ``events`` must contain ``event_file`` and ``event_segment`` columns. The
    returned waveforms preserve the input row order.
    """

    if len(events) == 0:
        raise ValueError("No events were requested")

    required = {"event_file", "event_segment"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    event_table = events.loc[:, ["event_file", "event_segment"]].copy()
    event_table["_request_order"] = np.arange(len(event_table))

    time_ns_ref = None
    waveform_parts = []
    order_parts = []

    for event_file, group in event_table.groupby("event_file", sort=False):
        segments = group["event_segment"].to_numpy()
        time_s, voltage_V, metadata = read_keysight_h5_direct(
            event_file,
            channel=channel,
            segment_numbers=segments,
        )
        time_ns, voltage_mV, *_ = standard_units(time_s, voltage_V, metadata)
        if time_origin == "zero":
            time_ns = time_ns - time_ns[0]
        elif time_origin != "original":
            raise ValueError("time_origin must be 'original' or 'zero'")

        if time_ns_ref is None:
            time_ns_ref = time_ns
        elif len(time_ns_ref) != len(time_ns):
            raise ValueError(f"Number of samples changed in {event_file}")

        if subtract_baseline:
            voltage_mV, _ = baseline_subtraction(
                time_ns_ref,
                voltage_mV,
                baseline_window_ns=baseline_window_ns,
                baseline_reference_time_ns=baseline_reference_time_ns,
                baseline_reference_mV=baseline_reference_mV,
            )
        waveform_parts.append(voltage_mV)
        order_parts.append(group["_request_order"].to_numpy())

    waveforms = np.concatenate(waveform_parts, axis=0)
    order = np.concatenate(order_parts)
    return time_ns_ref, waveforms[np.argsort(order)]





__all__ = [
    "extract_pmt_voltage",
    "load_acquisition_timing_exposure",
    "find_pmt_files",
    "load_baseline_reference",
    "load_preprocessed_waveforms",
    "read_waveform_sample",
    "resolve_baseline_reference_path",
    "load_files_streaming",
    "load_files_one_go",
    "integrate_event_fixed_window_charge",
    "stream_event_qdc_diagnostics",
    "resolve_cached_event_files",
    "load_event_waveforms",
]
