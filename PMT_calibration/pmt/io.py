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

    When ``skip_corrupt_files`` is true, HDF5 read errors skip the affected
    acquisition segment with an explicit warning while processing continues.
    """
    files = list(files)
    df_parts = []
    waveform_sample_parts = []
    time_ns_ref = None

    total_events = 0
    total_kept_after_saturation = 0
    total_kept_after_raw_baseline_check = 0
    total_kept_after_baseline_cut = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    current_file = None
    files_started = 0
    skipped_files = []

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


def load_event_waveforms(
    events,
    *,
    channel="Channel 1",
    baseline_window_ns=(0, 20),
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
    time_origin="zero",
):
    """Load and baseline-subtract specific events listed in a feature dataframe.

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
    "find_pmt_files",
    "load_baseline_reference",
    "load_preprocessed_waveforms",
    "read_waveform_sample",
    "resolve_baseline_reference_path",
    "load_files_streaming",
    "load_files_one_go",
    "load_event_waveforms",
]
