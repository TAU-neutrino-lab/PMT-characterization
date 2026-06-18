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

def load_baseline_subtracted_waveforms(
    files: Sequence[str | Path],
    baseline_window_ns,
    subtract_baseline,
    *,
    channel: str = "Channel 1",
    chunk_size: int = 512,
) -> dict[str, np.ndarray]:
    """Read PMT files and return baseline-subtracted waveforms.
    """

    voltage_parts = []
    baseline_parts = []
    event_files = []
    event_segments = []
    time_ns = None

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        if time_ns is None:
            time_ns = chunk["time_ns"]
        elif len(time_ns) != len(chunk["time_ns"]):
            raise ValueError(f"Number of samples changed in {chunk['filename']}")

        voltage_bs_mV, baseline_mV = subtract_baseline(
            chunk["time_ns"],
            chunk["voltage_mV"],
            baseline_window_ns=baseline_window_ns,
        )

        voltage_parts.append(voltage_bs_mV)
        baseline_parts.append(baseline_mV)
        event_files.extend([chunk["filename"]] * len(voltage_bs_mV))
        event_segments.extend(chunk["segment_numbers"])

    if time_ns is None:
        raise ValueError("No waveforms were loaded")

    return {
        "time_ns": time_ns,
        "voltage_bs_mV": np.concatenate(voltage_parts, axis=0),
        "baseline_mV": np.concatenate(baseline_parts),
        "event_file": np.array(event_files),
        "event_segment": np.array(event_segments),
    }


__all__ = [
    "find_pmt_files",
    "load_baseline_subtracted_waveforms",
    "read_waveform_sample",
]
