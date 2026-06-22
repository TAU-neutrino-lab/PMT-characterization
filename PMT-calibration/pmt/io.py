"""PMT waveform loading helpers.

This module is a PMT-facing layer over :mod:`lab_tools.io`. Keep generic
oscilloscope file parsing in ``lab_tools``; put PMT workflow conveniences here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


from lab_tools.io import (
    iter_keysight_chunks,
    read_keysight_h5_direct,
    read_segment_time_tags,
    standard_units,
)

from .preprocessing import *


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

    return {
        "time_ns": np.concatenate(all_time, axis=0),
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
    baseline_subtraction=True,
    baseline_window_ns=(0.0, 20.0),
    low_limit_mV=None,
    high_limit_mV=None,
    margin_mV=0.0,
    max_saturated_samples=0,
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
            time_ns = chunk["time_ns"]

        elif len(time_ns) != len(chunk["time_ns"]):
            raise ValueError(
                f"Number of samples changed in {chunk['filename']}"
            )

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

            voltage_mV, baseline_event_info = (
                subtract_baseline(
                    chunk["time_ns"],
                    voltage_mV,
                    baseline_window_ns=baseline_window_ns,
                )
            )

        else:
            baseline_event_info = {}

        # ---------- Merge event-level metadata ----------

        chunk_event_info = {}
        chunk_event_info.update( saturation_event_info)
        chunk_event_info.update( baseline_event_info)

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


__all__ = [
    "find_pmt_files",
    "load_preprocessed_waveforms",
    "read_waveform_sample",
]
