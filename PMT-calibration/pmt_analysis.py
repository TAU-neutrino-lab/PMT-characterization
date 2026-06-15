"""Reusable PMT LED calibration helpers.

The charge convention used here is:
    charge_mV_ns = - integral(baseline-subtracted voltage_mV dt_ns)

This makes negative-going PMT pulses have positive charge, with units mV ns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import re
from pathlib import Path

import h5py
import numpy as np

from scipy.optimize import curve_fit
from scipy.special import erfc, gammaln



# ----------------------------------------------------------------------------------------------
# Read files
# ----------------------------------------------------------------------------------------------

SEGMENT_RE = re.compile(r"Seg(\d+)Data$")

def segment_number(name: str) -> int | None:
    match = SEGMENT_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def sorted_segment_keys(channel_group) -> list[str]:
    keys = [key for key in channel_group.keys() if segment_number(key) is not None]
    return sorted(keys, key=segment_number)


def keysight_waveform_dataset(channel_group, channel: str = "Channel 1"):
    """Return the single Keysight waveform dataset used by non-segment groups."""
    data_name = f"{channel}Data"
    if data_name in channel_group:
        return channel_group[data_name]

    dataset_keys = [
        key
        for key, value in channel_group.items()
        if isinstance(value, h5py.Dataset) and key.endswith("Data") and segment_number(key) is None
    ]
    if len(dataset_keys) == 1:
        return channel_group[dataset_keys[0]]

    return None


def keysight_waveform_count(channel_group, waveform_dataset=None) -> int:
    attrs = channel_group.attrs
    if "NumWaveforms" in attrs:
        return int(attrs["NumWaveforms"])
    if "NumSegments" in attrs and int(attrs["NumSegments"]) > 0:
        return int(attrs["NumSegments"])
    if waveform_dataset is not None:
        n_samples = int(attrs["NumPoints"])
        shape = waveform_dataset.shape
        if len(shape) == 2:
            return int(shape[0] if shape[1] == n_samples else shape[1])
        if len(shape) == 1:
            return int(shape[0] // n_samples)
    return len(sorted_segment_keys(channel_group))


def read_keysight_waveform_rows(waveform_dataset, start: int, stop: int, n_samples: int, dtype=np.float32):
    """Read rows from a Keysight ``Channel NData`` dataset as ``(events, samples)``."""
    shape = waveform_dataset.shape
    if len(shape) == 2:
        if shape[1] == n_samples:
            return waveform_dataset[start:stop, :].astype(dtype, copy=False)
        if shape[0] == n_samples:
            return waveform_dataset[:, start:stop].T.astype(dtype, copy=False)
        raise ValueError(f"Cannot infer waveform axis from dataset shape {shape} and NumPoints={n_samples}")

    if len(shape) == 1:
        raw = waveform_dataset[start * n_samples : stop * n_samples]
        return raw.reshape(stop - start, n_samples).astype(dtype, copy=False)

    raise ValueError(f"Unsupported Keysight waveform dataset shape {shape}")


def keysight_channel_group(h5_file, channel: str = "Channel 1"):
    """Return the Keysight channel group for root-level or Waveforms layouts."""
    if channel in h5_file:
        return h5_file[channel]
    if "Waveforms" in h5_file and channel in h5_file["Waveforms"]:
        return h5_file["Waveforms"][channel]
    raise KeyError(f"Could not find {channel!r} in {h5_file.filename}")


def keysight_time_axis_ns(channel_group) -> np.ndarray:
    attrs = channel_group.attrs
    n_samples = int(attrs["NumPoints"])
    return (float(attrs["XOrg"]) + np.arange(n_samples) * float(attrs["XInc"])) * 1e9


def read_keysight_h5_direct(
    filename,
    channel: str = "Channel 1",
    segment_numbers=None,
    dtype=np.float32,
):
    """Read a Keysight HDF5 file into voltage in V.

    This replaces the lab-specific ``lab_tools.io.read_keysight_h5`` helper and
    returns ``time_s, voltage_V, metadata``. It supports both layouts seen in
    Keysight files:

    - one dataset per segment, e.g. ``Seg1Data``;
    - one waveform table, e.g. ``Channel 1Data``.
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as h5_file:
        channel_group = keysight_channel_group(h5_file, channel=channel)
        attrs = channel_group.attrs
        keys = sorted_segment_keys(channel_group)
        waveform_dataset = keysight_waveform_dataset(channel_group, channel=channel)

        n_samples = int(attrs["NumPoints"])
        time_s = keysight_time_axis_ns(channel_group) * 1e-9
        yinc = float(attrs["YInc"])
        yorg = float(attrs["YOrg"])

        if keys:
            if segment_numbers is not None:
                wanted = set(segment_numbers)
                keys = [key for key in keys if segment_number(key) in wanted]

            voltage = np.empty((len(keys), len(time_s)), dtype=dtype)
            for i, key in enumerate(keys):
                voltage[i] = channel_group[key][()] * yinc + yorg
            returned_segments = [segment_number(key) for key in keys]
            layout = "segment_datasets"
        elif waveform_dataset is not None:
            n_waveforms = keysight_waveform_count(channel_group, waveform_dataset)
            if segment_numbers is None:
                row_indices = np.arange(n_waveforms, dtype=int)
            else:
                row_indices = np.array(list(segment_numbers), dtype=int)
                row_indices = row_indices[(row_indices >= 0) & (row_indices < n_waveforms)]

            if len(row_indices) == 0:
                voltage = np.empty((0, len(time_s)), dtype=dtype)
            elif np.array_equal(row_indices, np.arange(row_indices[0], row_indices[-1] + 1)):
                voltage = read_keysight_waveform_rows(
                    waveform_dataset,
                    int(row_indices[0]),
                    int(row_indices[-1]) + 1,
                    n_samples,
                    dtype=dtype,
                )
                voltage = voltage * yinc + yorg
            else:
                voltage = np.empty((len(row_indices), len(time_s)), dtype=dtype)
                for i, row_index in enumerate(row_indices):
                    voltage[i] = read_keysight_waveform_rows(
                        waveform_dataset,
                        int(row_index),
                        int(row_index) + 1,
                        n_samples,
                        dtype=dtype,
                    )[0] * yinc + yorg
            returned_segments = row_indices.tolist()
            layout = "waveform_dataset"
        else:
            raise KeyError(f"Could not find segment datasets or a waveform dataset in {h5_file.filename}")

        metadata = {
            "filename": str(filename),
            "channel": channel,
            "channel_attrs": dict(attrs.items()),
            "segment_numbers": returned_segments,
            "layout": layout,
        }

    return time_s, voltage, metadata


def iter_keysight_chunks(files, channel: str = "Channel 1", chunk_size: int = 512, dtype=np.float32):
    """Yield waveform chunks in mV for one or more Keysight HDF5 files."""
    reference_len = None

    for filename in files:
        filename = Path(filename)
        with h5py.File(filename, "r") as h5_file:
            channel_group = keysight_channel_group(h5_file, channel=channel)
            attrs = channel_group.attrs
            time_ns = keysight_time_axis_ns(channel_group)

            if reference_len is None:
                reference_len = len(time_ns)
            elif len(time_ns) != reference_len:
                raise ValueError(f"Time axes have different lengths in {filename}")

            yinc_mV = float(attrs["YInc"]) * 1e3
            yorg_mV = float(attrs["YOrg"]) * 1e3
            keys = sorted_segment_keys(channel_group)
            waveform_dataset = keysight_waveform_dataset(channel_group, channel=channel)

            if keys:
                for start in range(0, len(keys), chunk_size):
                    chunk_keys = keys[start : start + chunk_size]
                    raw = np.stack([channel_group[key][()] for key in chunk_keys]).astype(dtype, copy=False)
                    voltage_mV = raw * yinc_mV + yorg_mV
                    yield {
                        "filename": str(filename),
                        "time_ns": time_ns,
                        "voltage_mV": voltage_mV,
                        "segment_numbers": [segment_number(key) for key in chunk_keys],
                        "metadata": {
                            "filename": str(filename),
                            "channel": channel,
                            "channel_attrs": dict(attrs.items()),
                            "layout": "segment_datasets",
                        },
                    }
            elif waveform_dataset is not None:
                n_samples = int(attrs["NumPoints"])
                n_waveforms = keysight_waveform_count(channel_group, waveform_dataset)
                for start in range(0, n_waveforms, chunk_size):
                    stop = min(start + chunk_size, n_waveforms)
                    raw = read_keysight_waveform_rows(waveform_dataset, start, stop, n_samples, dtype=dtype)
                    voltage_mV = raw * yinc_mV + yorg_mV
                    yield {
                        "filename": str(filename),
                        "time_ns": time_ns,
                        "voltage_mV": voltage_mV,
                        "segment_numbers": list(range(start, stop)),
                        "metadata": {
                            "filename": str(filename),
                            "channel": channel,
                            "channel_attrs": dict(attrs.items()),
                            "layout": "waveform_dataset",
                        },
                    }
            else:
                raise KeyError(f"Could not find segment datasets or a waveform dataset in {h5_file.filename}")

def standard_units(time, voltage, metadata):
    """Convert a direct-reader output from seconds/volts to ns/mV."""
    voltage_mV = voltage * 1e3
    adc_step_mV = metadata["channel_attrs"]["YInc"] * 1e3
    time_ns = time * 1e9
    return time_ns, voltage_mV, adc_step_mV, "ns", "mV"

# ----------------------------------------------------------------------------------------------
# Baseline Subtraction
# ----------------------------------------------------------------------------------------------

def subtract_baseline(time_ns, voltage_mV, baseline_window_ns=None, baseline_window=None):
    """Subtract each waveform baseline measured in ``baseline_window_ns``."""
    if baseline_window_ns is None:
        baseline_window_ns = baseline_window
    if baseline_window_ns is None:
        baseline_window_ns = (0.0, 20.0)

    mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if not np.any(mask):
        raise ValueError(f"Baseline window {baseline_window_ns} contains no samples")

    baseline_mV = np.mean(voltage_mV[:, mask], axis=1)
    voltage_bs_mV = voltage_mV - baseline_mV[:, None]
    return voltage_bs_mV, baseline_mV

# ----------------------------------------------------------------------------------------------
# Saturated wavefroms
# ----------------------------------------------------------------------------------------------

def saturated_waveform_mask(
    voltage_mV,
    low_limit_mV=None,
    high_limit_mV=None,
    margin_mV: float = 0.0,
    max_saturated_samples: int = 0,
):
    """Return a mask for raw waveforms that touch saturation limits.

    Run this before baseline subtraction. Set one or both voltage limits. For
    example, with ``low_limit_mV=-30`` and ``margin_mV=0.2``, samples at or
    below ``-29.8 mV`` count as saturated.
    """
    if low_limit_mV is None and high_limit_mV is None:
        raise ValueError("Provide at least one of low_limit_mV or high_limit_mV")

    voltage_mV = np.asarray(voltage_mV)
    saturated_samples = np.zeros(voltage_mV.shape, dtype=bool)

    if low_limit_mV is not None:
        saturated_samples |= voltage_mV <= (low_limit_mV + margin_mV)
    if high_limit_mV is not None:
        saturated_samples |= voltage_mV >= (high_limit_mV - margin_mV)

    n_saturated_samples = np.sum(saturated_samples, axis=1)
    saturated = n_saturated_samples > max_saturated_samples
    return saturated, n_saturated_samples


def remove_saturated_waveforms(
    voltage_mV,
    low_limit_mV=None,
    high_limit_mV=None,
    margin_mV: float = 0.0,
    max_saturated_samples: int = 0,
):
    """Remove saturated raw waveforms and return ``selected, keep_mask, info``."""
    saturated, n_saturated_samples = saturated_waveform_mask(
        voltage_mV,
        low_limit_mV=low_limit_mV,
        high_limit_mV=high_limit_mV,
        margin_mV=margin_mV,
        max_saturated_samples=max_saturated_samples,
    )
    keep_mask = ~saturated
    info = {
        "saturated_mask": saturated,
        "n_saturated_samples": n_saturated_samples,
        "n_total": int(len(keep_mask)),
        "n_kept": int(np.sum(keep_mask)),
        "n_removed": int(np.sum(saturated)),
        "efficiency": float(np.mean(keep_mask)) if len(keep_mask) else np.nan,
    }
    return voltage_mV[keep_mask], keep_mask, info


def _metadata_voltage_limits_mV(metadata):
    attrs = metadata.get("channel_attrs", metadata)
    if "YDispOrigin" not in attrs or "YDispRange" not in attrs:
        return None, None
    center_mV = float(attrs["YDispOrigin"]) * 1e3
    half_range_mV = float(attrs["YDispRange"]) * 1e3 / 2.0
    return center_mV - half_range_mV, center_mV + half_range_mV


def collect_saturation_selection(
    files,
    low_limit_mV=None,
    high_limit_mV=None,
    margin_mV: float = 0.0,
    max_saturated_samples: int = 0,
    infer_limits_from_metadata: bool = False,
    channel: str = "Channel 1",
    chunk_size: int = 512,
):
    """Build a whole-file keep mask that removes saturated raw waveforms.

    The returned ``keep_mask`` can be passed as ``selection_mask`` to the later
    baseline, timing, charge, or scan helpers.
    """
    keep_parts = []
    saturated_parts = []
    saturated_sample_parts = []
    used_low_limit_mV = low_limit_mV
    used_high_limit_mV = high_limit_mV

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        chunk_low_limit_mV = used_low_limit_mV
        chunk_high_limit_mV = used_high_limit_mV
        if infer_limits_from_metadata and (chunk_low_limit_mV is None or chunk_high_limit_mV is None):
            inferred_low, inferred_high = _metadata_voltage_limits_mV(chunk["metadata"])
            if chunk_low_limit_mV is None:
                chunk_low_limit_mV = inferred_low
                used_low_limit_mV = inferred_low
            if chunk_high_limit_mV is None:
                chunk_high_limit_mV = inferred_high
                used_high_limit_mV = inferred_high

        saturated, n_saturated_samples = saturated_waveform_mask(
            chunk["voltage_mV"],
            low_limit_mV=chunk_low_limit_mV,
            high_limit_mV=chunk_high_limit_mV,
            margin_mV=margin_mV,
            max_saturated_samples=max_saturated_samples,
        )
        saturated_parts.append(saturated)
        saturated_sample_parts.append(n_saturated_samples)
        keep_parts.append(~saturated)

    keep_mask = np.concatenate(keep_parts)
    saturated_mask = np.concatenate(saturated_parts)
    n_saturated_samples = np.concatenate(saturated_sample_parts)
    return {
        "keep_mask": keep_mask,
        "saturated_mask": saturated_mask,
        "n_saturated_samples": n_saturated_samples,
        "n_total": int(len(keep_mask)),
        "n_kept": int(np.sum(keep_mask)),
        "n_removed": int(np.sum(saturated_mask)),
        "efficiency": float(np.mean(keep_mask)) if len(keep_mask) else np.nan,
        "low_limit_mV": used_low_limit_mV,
        "high_limit_mV": used_high_limit_mV,
        "margin_mV": margin_mV,
        "max_saturated_samples": max_saturated_samples,
    }

# ----------------------------------------------------------------------------------------------
# Compute charge
# ----------------------------------------------------------------------------------------------

def compute_charge_window(time_ns, voltage_bs_mV, window_ns=None, tmin_ns=None, tmax_ns=None):
    """Integrate a fixed window and return positive charge for negative pulses."""
    if window_ns is not None:
        tmin_ns, tmax_ns = window_ns
    if tmin_ns is None or tmax_ns is None:
        raise ValueError("Provide either window_ns=(tmin, tmax) or tmin_ns/tmax_ns")

    mask = (time_ns >= tmin_ns) & (time_ns <= tmax_ns)
    if not np.any(mask):
        raise ValueError(f"Integration window {(tmin_ns, tmax_ns)} contains no samples")
    area_mV_ns = np.trapezoid(voltage_bs_mV[:, mask], x=time_ns[mask], axis=1)
    return -area_mV_ns


def compute_charges(time_ns, voltage_bs_mV, integration_window_ns):
    """Compute charge for the provided waveforms in a fixed integration window.

    This function is intentionally selection-agnostic: apply any event selection
    before calling it, then pass the selected baseline-subtracted waveforms here.
    """
    return compute_charge_window(time_ns, voltage_bs_mV, window_ns=integration_window_ns)


def compute_charge_around_minimum(time_ns, voltage_bs_mV, pre_samples: int = 20, post_samples: int = 40):
    """Integrate around each waveform minimum and return positive charge for negative pulses."""
    idx_min = np.argmin(voltage_bs_mV, axis=1)
    charge = np.zeros(len(voltage_bs_mV), dtype=np.float64)

    for i, idx in enumerate(idx_min):
        start = max(0, idx - pre_samples)
        stop = min(len(time_ns), idx + post_samples)
        charge[i] = -np.trapezoid(voltage_bs_mV[i, start:stop], x=time_ns[start:stop])

    return charge


def _window_mask(time_ns, window_ns):
    mask = (time_ns >= window_ns[0]) & (time_ns <= window_ns[1])
    if not np.any(mask):
        raise ValueError(f"Window {window_ns} contains no samples")
    return mask


def compute_charges_for_window(
    files,
    baseline_window_ns,
    integration_window_ns,
    selection_mask=None,
    channel: str = "Channel 1",
    chunk_size: int = 512,
):
    """Compute charges for one fixed window, optionally applying a precomputed mask."""
    return compute_charges_for_windows(
        files,
        baseline_window_ns,
        [integration_window_ns],
        selection_mask=selection_mask,
        channel=channel,
        chunk_size=chunk_size,
    )[tuple(map(float, integration_window_ns))]


def compute_charges_for_windows(
    files,
    baseline_window_ns,
    integration_windows_ns,
    selection_mask=None,
    channel: str = "Channel 1",
    chunk_size: int = 512,
):
    """Compute charges for many fixed windows in one pass through the files.

    ``selection_mask`` is optional and must be a boolean array in the same event
    order yielded by ``iter_keysight_chunks``. Selection is deliberately external
    to this function: this helper only applies a mask it is given.
    """
    windows = [tuple(map(float, window)) for window in integration_windows_ns]
    charge_parts = {window: [] for window in windows}
    selection_mask = _prepare_selection_mask(selection_mask)
    selection_offset = 0

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        time_ns = chunk["time_ns"]
        voltage_bs_mV, _ = subtract_baseline(
            time_ns,
            chunk["voltage_mV"],
            baseline_window_ns=baseline_window_ns,
        )

        # Cumulative trapezoid integral lets each fixed window become one
        # subtraction rather than a fresh integration over all samples.
        dt = np.diff(time_ns)
        trap = 0.5 * (voltage_bs_mV[:, :-1] + voltage_bs_mV[:, 1:]) * dt
        cumulative = np.concatenate(
            [np.zeros((voltage_bs_mV.shape[0], 1), dtype=np.float64), np.cumsum(trap, axis=1)],
            axis=1,
        )
        if selection_mask is None:
            chunk_selection = slice(None)
        else:
            chunk_selection, selection_offset = _selection_for_chunk(
                selection_mask,
                selection_offset,
                voltage_bs_mV.shape[0],
            )

        for window in windows:
            start = int(np.searchsorted(time_ns, window[0], side="left"))
            stop = int(np.searchsorted(time_ns, window[1], side="right") - 1)
            if stop <= start:
                raise ValueError(f"Integration window {window} contains too few samples")
            charge_parts[window].append(-(cumulative[chunk_selection, stop] - cumulative[chunk_selection, start]))

    _check_selection_consumed(selection_mask, selection_offset)

    return {window: np.concatenate(parts) for window, parts in charge_parts.items()}

def scan_integration_windows(
    files,
    baseline_window_ns,
    starts_ns,
    stops_ns,
    channel: str = "Channel 1",
    chunk_size: int = 512,
    min_width_ns: float = 1.0,
    selection_mask=None,
):
    windows = [(float(start), float(stop)) for start in starts_ns for stop in stops_ns if stop - start >= min_width_ns]
    charges_by_window = compute_charges_for_windows(
        files,
        baseline_window_ns,
        windows,
        selection_mask=selection_mask,
        channel=channel,
        chunk_size=chunk_size,
    )

    results = []
    for window, charge in charges_by_window.items():
        metrics = score_charge_spectrum(charge)
        results.append(
            {
                "window_ns": window,
                "charge_mV_ns": charge,
                **metrics,
            }
        )
    return sorted(results, key=lambda row: row["score"], reverse=True)

# ----------------------------------------------------------------------------------------------
# Properties and selection
# ----------------------------------------------------------------------------------------------

def peak_properties(time_ns, voltage_bs_mV, search_window_ns=None, polarity: str = "negative"):
    """Measure peak time, amplitude, and sample index for each waveform.

    For the default ``polarity='negative'``, amplitude is ``-V_min`` so real PMT
    pulses have positive amplitude.
    """
    if search_window_ns is None:
        search_mask = np.ones_like(time_ns, dtype=bool)
    else:
        search_mask = _window_mask(time_ns, search_window_ns)

    search_indices = np.flatnonzero(search_mask)
    search_voltage = voltage_bs_mV[:, search_mask]

    if polarity == "negative":
        local_peak_index = np.argmin(search_voltage, axis=1)
        peak_value_mV = search_voltage[np.arange(len(search_voltage)), local_peak_index]
        peak_amplitude_mV = -peak_value_mV
    elif polarity == "positive":
        local_peak_index = np.argmax(search_voltage, axis=1)
        peak_value_mV = search_voltage[np.arange(len(search_voltage)), local_peak_index]
        peak_amplitude_mV = peak_value_mV
    else:
        raise ValueError("polarity must be 'negative' or 'positive'")

    peak_index = search_indices[local_peak_index]
    return {
        "peak_index": peak_index,
        "peak_time_ns": time_ns[peak_index],
        "peak_value_mV": peak_value_mV,
        "peak_amplitude_mV": peak_amplitude_mV,
    }


def rise_time_properties(
    time_ns,
    voltage_bs_mV,
    search_window_ns=None,
    low_fraction: float = 0.1,
    high_fraction: float = 0.9,
    polarity: str = "negative",
):
    """Measure leading-edge rise times between two peak-amplitude fractions.

    The crossing times are found by linear interpolation on the leading edge.
    For negative PMT pulses the returned amplitude is positive.
    """
    peaks = peak_properties(time_ns, voltage_bs_mV, search_window_ns=search_window_ns, polarity=polarity)
    n_events = voltage_bs_mV.shape[0]
    t_low = np.full(n_events, np.nan, dtype=float)
    t_high = np.full(n_events, np.nan, dtype=float)

    for i in range(n_events):
        peak_idx = int(peaks["peak_index"][i])
        amplitude = peaks["peak_amplitude_mV"][i]
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue

        if polarity == "negative":
            signal = -voltage_bs_mV[i, : peak_idx + 1]
        else:
            signal = voltage_bs_mV[i, : peak_idx + 1]
        t = time_ns[: peak_idx + 1]

        for fraction, output in ((low_fraction, t_low), (high_fraction, t_high)):
            threshold = fraction * amplitude
            crossing_indices = np.flatnonzero(signal >= threshold)
            if len(crossing_indices) == 0:
                continue
            idx = int(crossing_indices[0])
            if idx == 0:
                output[i] = t[0]
                continue
            y0, y1 = signal[idx - 1], signal[idx]
            x0, x1 = t[idx - 1], t[idx]
            if y1 == y0:
                output[i] = x1
            else:
                output[i] = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)

    rise_time = t_high - t_low
    valid = np.isfinite(rise_time) & (rise_time >= 0)
    return {
        **peaks,
        "t_low_ns": t_low,
        "t_high_ns": t_high,
        "rise_time_ns": rise_time,
        "rise_time_valid": valid,
        "low_fraction": low_fraction,
        "high_fraction": high_fraction,
    }


def peak_window_selection(
    time_ns,
    voltage_bs_mV,
    allowed_peak_window_ns,
    search_window_ns=None,
    min_amplitude_mV=None,
    polarity: str = "negative",
):
    """Return a boolean mask for waveforms whose peak is in a chosen time window."""
    peaks = peak_properties(time_ns, voltage_bs_mV, search_window_ns=search_window_ns, polarity=polarity)
    mask = (peaks["peak_time_ns"] >= allowed_peak_window_ns[0]) & (peaks["peak_time_ns"] <= allowed_peak_window_ns[1])
    if min_amplitude_mV is not None:
        mask &= peaks["peak_amplitude_mV"] >= min_amplitude_mV
    return mask, peaks


def peak_time_voltage_selection(
    time_ns,
    voltage_bs_mV,
    peak_window_ns,
    global_peak_voltage_max_mV=None,
    out_of_window_peak_voltage_max_mV=None,
    search_window_ns=None,
    polarity: str = "negative",
):
    """Select events with a global threshold and an out-of-time pulse veto.

    For negative PMT pulses:
    - ``global_peak_voltage_max_mV=-X`` removes events whose minimum is larger
      than -X mV, i.e. keeps only ``peak_value_mV <= -X``. Set to ``None`` to
      keep pedestal events.
    - ``out_of_window_peak_voltage_max_mV=-Y`` rejects real pulses outside
      ``peak_window_ns`` only when ``peak_value_mV <= -Y``. This keeps pedestal
      events whose noise minimum happens outside the timing window.
    """
    peaks = peak_properties(time_ns, voltage_bs_mV, search_window_ns=search_window_ns, polarity=polarity)
    return peak_time_voltage_selection_from_properties(
        peaks,
        peak_window_ns=peak_window_ns,
        global_peak_voltage_max_mV=global_peak_voltage_max_mV,
        out_of_window_peak_voltage_max_mV=out_of_window_peak_voltage_max_mV,
        polarity=polarity,
    ), peaks


def peak_time_voltage_selection_from_properties(
    peaks,
    peak_window_ns,
    global_peak_voltage_max_mV=None,
    out_of_window_peak_voltage_max_mV=None,
    polarity: str = "negative",
):
    """Apply peak timing/voltage selection to precomputed peak properties."""
    in_time = (peaks["peak_time_ns"] >= peak_window_ns[0]) & (peaks["peak_time_ns"] <= peak_window_ns[1])

    if polarity == "negative":
        keep = np.ones_like(in_time, dtype=bool)
        if global_peak_voltage_max_mV is not None:
            keep &= peaks["peak_value_mV"] <= global_peak_voltage_max_mV
        if out_of_window_peak_voltage_max_mV is not None:
            out_of_time_real_peak = (~in_time) & (peaks["peak_value_mV"] <= out_of_window_peak_voltage_max_mV)
            keep &= ~out_of_time_real_peak
    elif polarity == "positive":
        keep = np.ones_like(in_time, dtype=bool)
        if global_peak_voltage_max_mV is not None:
            keep &= peaks["peak_value_mV"] >= global_peak_voltage_max_mV
        if out_of_window_peak_voltage_max_mV is not None:
            out_of_time_real_peak = (~in_time) & (peaks["peak_value_mV"] >= out_of_window_peak_voltage_max_mV)
            keep &= ~out_of_time_real_peak
    else:
        raise ValueError("polarity must be 'negative' or 'positive'")

    return keep


def real_peak_mask_from_properties(peaks, peak_voltage_max_mV, polarity: str = "negative"):
    """Return events whose peak voltage passes a real-pulse threshold."""
    if polarity == "negative":
        return peaks["peak_value_mV"] <= peak_voltage_max_mV
    if polarity == "positive":
        return peaks["peak_value_mV"] >= peak_voltage_max_mV
    raise ValueError("polarity must be 'negative' or 'positive'")


def select_waveforms_by_peak_time_and_voltage(
    time_ns,
    voltage_bs_mV,
    peak_window_ns,
    global_peak_voltage_max_mV=None,
    out_of_window_peak_voltage_max_mV=None,
    search_window_ns=None,
    polarity: str = "negative",
):
    """Return waveforms passing the peak timing/voltage selection."""
    mask, peaks = peak_time_voltage_selection(
        time_ns,
        voltage_bs_mV,
        peak_window_ns=peak_window_ns,
        global_peak_voltage_max_mV=global_peak_voltage_max_mV,
        out_of_window_peak_voltage_max_mV=out_of_window_peak_voltage_max_mV,
        search_window_ns=search_window_ns,
        polarity=polarity,
    )
    return voltage_bs_mV[mask], mask, peaks


def select_waveforms_by_peak_window(
    time_ns,
    voltage_bs_mV,
    allowed_peak_window_ns,
    search_window_ns=None,
    min_amplitude_mV=None,
    polarity: str = "negative",
):
    """Select in-memory waveforms that peak inside ``allowed_peak_window_ns``."""
    mask, peaks = peak_window_selection(
        time_ns,
        voltage_bs_mV,
        allowed_peak_window_ns=allowed_peak_window_ns,
        search_window_ns=search_window_ns,
        min_amplitude_mV=min_amplitude_mV,
        polarity=polarity,
    )
    return voltage_bs_mV[mask], mask, peaks


def _prepare_selection_mask(selection_mask):
    return None if selection_mask is None else np.asarray(selection_mask, dtype=bool)


def _selection_for_chunk(selection_mask, offset, n_chunk):
    if selection_mask is None:
        return slice(None), offset

    chunk_selection = selection_mask[offset : offset + n_chunk]
    if len(chunk_selection) != n_chunk:
        raise ValueError("selection_mask is shorter than the number of events in files")
    return chunk_selection, offset + n_chunk


def _check_selection_consumed(selection_mask, offset):
    if selection_mask is not None and offset != len(selection_mask):
        raise ValueError("selection_mask is longer than the number of events in files")


def collect_peak_timing(
    files,
    baseline_window_ns,
    peak_search_window_ns=None,
    selection_mask=None,
    channel: str = "Channel 1",
    chunk_size: int = 512,
    low_fraction: float = 0.1,
    high_fraction: float = 0.9,
    polarity: str = "negative",
):
    """Collect peak-time, amplitude, and rise-time quantities for selected events."""
    parts = {
        "peak_time_ns": [],
        "peak_amplitude_mV": [],
        "peak_value_mV": [],
        "t_low_ns": [],
        "t_high_ns": [],
        "rise_time_ns": [],
        "rise_time_valid": [],
    }
    time_ns = None
    selection_mask = _prepare_selection_mask(selection_mask)
    selection_offset = 0

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        time_ns = chunk["time_ns"]
        chunk_selection, selection_offset = _selection_for_chunk(
            selection_mask,
            selection_offset,
            chunk["voltage_mV"].shape[0],
        )
        voltage_bs_mV, _ = subtract_baseline(time_ns, chunk["voltage_mV"], baseline_window_ns=baseline_window_ns)
        voltage_bs_mV = voltage_bs_mV[chunk_selection]
        if voltage_bs_mV.shape[0] == 0:
            continue
        timing = rise_time_properties(
            time_ns,
            voltage_bs_mV,
            search_window_ns=peak_search_window_ns,
            low_fraction=low_fraction,
            high_fraction=high_fraction,
            polarity=polarity,
        )
        for key in parts:
            parts[key].append(timing[key])

    _check_selection_consumed(selection_mask, selection_offset)
    if not parts["peak_time_ns"]:
        raise ValueError("No events left after applying selection_mask")
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result["time_ns"] = time_ns
    result["low_fraction"] = low_fraction
    result["high_fraction"] = high_fraction
    return result


def collect_peak_time_voltage_selection(
    files,
    baseline_window_ns,
    peak_window_ns,
    global_peak_voltage_max_mV=None,
    out_of_window_peak_voltage_max_mV=None,
    peak_search_window_ns=None,
    selection_mask=None,
    channel: str = "Channel 1",
    chunk_size: int = 512,
    low_fraction: float = 0.1,
    high_fraction: float = 0.9,
    polarity: str = "negative",
):
    """Collect a whole-file mask for the peak timing/voltage selection.

    If ``selection_mask`` is provided, the returned ``keep_mask`` is the logical
    AND of that preselection and this peak timing/voltage selection.
    """
    kept_masks = []
    real_peak_masks = []
    in_peak_window_masks = []
    peak_times = []
    peak_values = []
    peak_amplitudes = []
    t_low = []
    t_high = []
    rise_times = []
    rise_time_valid = []
    selection_mask = _prepare_selection_mask(selection_mask)
    selection_offset = 0

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        time_ns = chunk["time_ns"]
        chunk_preselection, selection_offset = _selection_for_chunk(
            selection_mask,
            selection_offset,
            chunk["voltage_mV"].shape[0],
        )
        voltage_bs_mV, _ = subtract_baseline(time_ns, chunk["voltage_mV"], baseline_window_ns=baseline_window_ns)
        timing = rise_time_properties(
            time_ns,
            voltage_bs_mV,
            search_window_ns=peak_search_window_ns,
            low_fraction=low_fraction,
            high_fraction=high_fraction,
            polarity=polarity,
        )
        keep = peak_time_voltage_selection_from_properties(
            timing,
            peak_window_ns=peak_window_ns,
            global_peak_voltage_max_mV=global_peak_voltage_max_mV,
            out_of_window_peak_voltage_max_mV=out_of_window_peak_voltage_max_mV,
            polarity=polarity,
        )
        if not isinstance(chunk_preselection, slice):
            keep = keep & chunk_preselection
        in_peak_window = (timing["peak_time_ns"] >= peak_window_ns[0]) & (timing["peak_time_ns"] <= peak_window_ns[1])
        if out_of_window_peak_voltage_max_mV is not None:
            real_peak = real_peak_mask_from_properties(
                timing,
                peak_voltage_max_mV=out_of_window_peak_voltage_max_mV,
                polarity=polarity,
            )
        else:
            real_peak = np.zeros_like(keep, dtype=bool)

        kept_masks.append(keep)
        real_peak_masks.append(real_peak)
        in_peak_window_masks.append(in_peak_window)
        peak_times.append(timing["peak_time_ns"])
        peak_values.append(timing["peak_value_mV"])
        peak_amplitudes.append(timing["peak_amplitude_mV"])
        t_low.append(timing["t_low_ns"])
        t_high.append(timing["t_high_ns"])
        rise_times.append(timing["rise_time_ns"])
        rise_time_valid.append(timing["rise_time_valid"])

    _check_selection_consumed(selection_mask, selection_offset)
    keep_mask = np.concatenate(kept_masks)
    return {
        "keep_mask": keep_mask, # True means the event passed the peak timing/voltage selection.
        "n_total": int(len(keep_mask)),
        "n_kept":  int(np.sum(keep_mask)),
        "efficiency": float(np.mean(keep_mask)) if len(keep_mask) else np.nan, # kept fraction, n_kept / n_total
        "real_peak_mask": np.concatenate(real_peak_masks), # boolean mask for events whose peak passes the “real pulse” voltage threshold (in the full sample)
        "in_peak_window_mask": np.concatenate(in_peak_window_masks), # boolean mask for events whose measured peak time is inside peak_window_ns
        "peak_time_ns":  np.concatenate(peak_times),  # measured peak time for each event
        "peak_value_mV": np.concatenate(peak_values), # signed voltage value at the peak. For negative PMT pulses this is usually negative
        "peak_amplitude_mV": np.concatenate(peak_amplitudes), # positive pulse amplitude. For negative pulses this is basically -peak_value_mV
        
        "t_low_ns":  np.concatenate(t_low),  # time where the rising edge first crosses low_fraction * peak_amplitude
        "t_high_ns": np.concatenate(t_high), # time where the rising edge first crosses high_fraction * peak_amplitude
        "rise_time_ns": np.concatenate(rise_times), # t_high_ns - t_low_ns
        "rise_time_valid": np.concatenate(rise_time_valid), # boolean mask saying whether the rise time was finite and non-negative
        "low_fraction":  low_fraction,  # the fractions used for the rise-time calculation (usually 0.1)
        "high_fraction": high_fraction, # the fractions used for the rise-time calculation (usually 0.9)
    }


def summarize_waveforms(files, baseline_window_ns, selection_mask=None, channel: str = "Channel 1", chunk_size: int = 512):
    """Compute baseline QA quantities and the average selected waveform."""
    n_events = 0
    mean_waveform = None
    m2_waveform = None
    baseline_values = []
    baseline_check_values = []
    pulse_min_times = []
    pulse_amplitudes = []
    time_ns = None
    selection_mask = _prepare_selection_mask(selection_mask)
    selection_offset = 0

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        time_ns = chunk["time_ns"]
        chunk_selection, selection_offset = _selection_for_chunk(
            selection_mask,
            selection_offset,
            chunk["voltage_mV"].shape[0],
        )
        voltage_bs_mV, baseline_mV = subtract_baseline(
            time_ns,
            chunk["voltage_mV"],
            baseline_window_ns=baseline_window_ns,
        )
        voltage_bs_mV = voltage_bs_mV[chunk_selection]
        baseline_mV = baseline_mV[chunk_selection]
        if voltage_bs_mV.shape[0] == 0:
            continue

        if mean_waveform is None:
            mean_waveform = np.zeros(voltage_bs_mV.shape[1], dtype=np.float64)
            m2_waveform = np.zeros_like(mean_waveform)

        n_chunk = voltage_bs_mV.shape[0]
        chunk_mean = np.mean(voltage_bs_mV, axis=0)
        chunk_m2 = np.sum((voltage_bs_mV - chunk_mean) ** 2, axis=0)
        delta = chunk_mean - mean_waveform
        n_total = n_events + n_chunk
        mean_waveform += delta * n_chunk / n_total
        m2_waveform += chunk_m2 + delta**2 * n_events * n_chunk / n_total
        n_events = n_total

        check_mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
        baseline_values.append(baseline_mV)
        baseline_check_values.append(np.mean(voltage_bs_mV[:, check_mask], axis=1))

        idx_min = np.argmin(voltage_bs_mV, axis=1)
        pulse_min_times.append(time_ns[idx_min])
        pulse_amplitudes.append(-voltage_bs_mV[np.arange(n_chunk), idx_min])

    _check_selection_consumed(selection_mask, selection_offset)
    if mean_waveform is None:
        raise ValueError("No events left after applying selection_mask")

    std_waveform = np.sqrt(m2_waveform / max(n_events - 1, 1))

    return {
        "n_events": n_events,
        "time_ns": time_ns,
        "mean_waveform_mV": mean_waveform,
        "std_waveform_mV": std_waveform,
        "baseline_mV": np.concatenate(baseline_values),
        "baseline_check_mV": np.concatenate(baseline_check_values),
        "pulse_min_time_ns": np.concatenate(pulse_min_times),
        "pulse_amplitude_mV": np.concatenate(pulse_amplitudes),
    }


def average_waveforms_by_amplitude(voltage_mV, amplitude_bins_mV):
    amplitude_mV = -np.min(voltage_mV, axis=1)
    results = []

    for low, high in zip(amplitude_bins_mV[:-1], amplitude_bins_mV[1:]):
        mask = (amplitude_mV >= low) & (amplitude_mV < high)
        n_events = int(np.sum(mask))

        if n_events > 0:
            average_waveform = np.mean(voltage_mV[mask], axis=0)
            mean_amplitude = np.mean(amplitude_mV[mask])
            std_amplitude = np.std(amplitude_mV[mask])
        else:
            average_waveform = None
            mean_amplitude = np.nan
            std_amplitude = np.nan

        results.append(
            {
                "bin": (low, high),
                "n_events": n_events,
                "mean_amplitude_mV": mean_amplitude,
                "std_amplitude_mV": std_amplitude,
                "average_waveform": average_waveform,
                "mask": mask,
            }
        )

    return results

# ----------------------------------------------------------------------------------------------
# Fits
# when possible, the notation is taken from Bellamy et. al's paper 
# absolute calibration and monitoring of a spectrometric channel using a photomultiplier
# ----------------------------------------------------------------------------------------------

# --------- Rough estimate fo initial parameters ---------

def score_charge_spectrum(charge_mV_ns, scale_mV_ns: float = 50.0):
    """Return simple window-quality metrics from a charge spectrum.

    The scale only improves numerical conditioning for the rough peak finding;
    returned SPE positions are converted back to mV ns.
    """
    charge_score = charge_mV_ns / scale_mV_ns
    hist_range = (np.quantile(charge_score, 0.001), np.quantile(charge_score, 0.995))
    counts, edges = np.histogram(charge_score, bins=240, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pedestal_idx = int(np.argmax(counts))
    pedestal_center = centers[pedestal_idx]
    pedestal_sigma = np.std(charge_score[np.abs(charge_score - pedestal_center) < 0.04])
    pedestal_sigma = max(pedestal_sigma, 0.012)

    search_mask = centers > pedestal_center + 2.0 * pedestal_sigma
    if np.any(search_mask):
        spe_rel_idx = int(np.argmax(counts[search_mask]))
        spe_center = centers[search_mask][spe_rel_idx]
        spe_height = counts[search_mask][spe_rel_idx]
    else:
        spe_center = np.nan
        spe_height = 0

    rough_spe_area = (spe_center - pedestal_center) * scale_mV_ns
    score = spe_height / np.sqrt(max(counts[pedestal_idx], 1))
    if not np.isfinite(rough_spe_area) or rough_spe_area <= 0:
        score = -np.inf

    return {
        "score": float(score),
        "rough_pedestal_mV_ns": float(pedestal_center * scale_mV_ns),
        "rough_spe_area_mV_ns": float(rough_spe_area),
        "rough_pedestal_sigma_mV_ns": float(pedestal_sigma * scale_mV_ns),
    }


# --------- PDFs ---------

def gaussian(x, amplitude, mean, sigma):
    sigma = np.maximum(sigma, 1e-9)
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def two_gaussian_plus_constant(x, a0, m0, s0, a1, m1, s1, c):
    return gaussian(x, a0, m0, s0) + gaussian(x, a1, m1, s1) + c


def gaussian_pdf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def exponential_gaussian_pdf(x, mean, sigma, alpha):
    """PDF of Gaussian(mean, sigma) convolved with alpha*exp(-alpha*x), x>=0."""
    sigma = np.maximum(sigma, 1e-12)
    alpha = np.maximum(alpha, 1e-12)
    z = (mean + alpha * sigma**2 - x) / (np.sqrt(2) * sigma)
    return 0.5 * alpha * np.exp(alpha * (mean - x) + 0.5 * (alpha * sigma) ** 2) * erfc(z)


def asymmetric_gaussian_pdf(x, mean, sigma_left, sigma_right):
    """Normalized split-normal PDF with different left/right widths."""
    sigma_left = np.maximum(sigma_left, 1e-9)
    sigma_right = np.maximum(sigma_right, 1e-9)
    norm = np.sqrt(2.0 / np.pi) / (sigma_left + sigma_right)
    sigma = np.where(x < mean, sigma_left, sigma_right)
    return norm * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# --------- Simple Poisson Fit ---------

def poisson_spe_parameter_names() -> list[str]:
    return [
        "n_total",
        "mu_pe",
        "q0_mV_ns",
        "sigma0_mV_ns",
        "q1_mV_ns",
        "sigma1_mV_ns",
    ]


def poisson_spe_model(
    x,
    n_total,
    mu_pe,
    q0,
    sigma0,
    q1,
    sigma1,
    bin_width,
    max_pe: int = 8,
):
    """Bellamy-style ideal Gaussian PM response.

    Parameter names follow Bellamy notation:
    Q0 -> ``q0_mV_ns``, sigma0 -> ``sigma0_mV_ns``,
    Q1 -> ``q1_mV_ns``, sigma1 -> ``sigma1_mV_ns``.
    """
    y = np.zeros_like(x, dtype=float)
    mu_pe = np.maximum(mu_pe, 1e-12)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = q0 + n_pe * q1
        sigma = np.sqrt(sigma0**2 + n_pe * sigma1**2)
        y += n_total * bin_width * weight * gaussian_pdf(x, mean, sigma)
    return y


def poisson_spe_components(
    x,
    parameters,
    bin_width,
    max_pe: int = 8,
):
    """Return the individual n-photoelectron Gaussian components."""
    components = {}
    p = parameters
    mu_pe = np.maximum(p["mu_pe"], 1e-12)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = p["q0_mV_ns"] + n_pe * p["q1_mV_ns"]
        sigma = np.sqrt(p["sigma0_mV_ns"] ** 2 + n_pe * p["sigma1_mV_ns"] ** 2)
        components[n_pe] = p["n_total"] * bin_width * weight * gaussian_pdf(x, mean, sigma)

    return components


def fit_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 8,
    p0=None,
    maxfev: int = 100000,
):
    """Fit the charge spectrum with the Poisson-weighted SPE model.

    ``p0`` must be ``{param_name: [initial, lower, upper, is_fixed]}``.
    """
    fit = fit_histogram_model(
        charge_mV_ns,
        poisson_spe_model,
        poisson_spe_parameter_names(),
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_name="poisson_spe",
        model_kwargs={"max_pe": max_pe},
        maxfev=maxfev,
    )
    fit["max_pe"] = max_pe
    return fit


# --------- AsymmetricPoisson Fit ---------

def free_spe_parameter_names(max_pe: int = 3) -> list[str]:
    return [
        "n_pedestal",
        "q0_mV_ns",
        "sigma0_mV_ns",
        "q1_mV_ns",
        "sigma1_left_mV_ns",
        "sigma1_right_mV_ns",
        *[f"n_{n_pe}pe" for n_pe in range(1, max_pe + 1)],
        "constant_per_bin",
    ]


def free_spe_model(
    x,
    n_pedestal,
    pedestal,
    sigma_ped,
    spe_area,
    sigma_spe_left,
    sigma_spe_right,
    *peak_areas_and_constant,
    bin_width,
    max_pe: int = 3,
):
    """Pedestal plus free-area asymmetric PE peaks.

    Unlike ``poisson_spe_model``, this does not constrain the PE peak areas to
    follow a Poisson law. That is often a better diagnostic model for dark-count
    or threshold-selected charge spectra.
    """
    if len(peak_areas_and_constant) != max_pe + 1:
        raise ValueError(f"Expected {max_pe + 1} trailing parameters, got {len(peak_areas_and_constant)}")

    peak_areas = peak_areas_and_constant[:max_pe]
    constant_per_bin = peak_areas_and_constant[-1]

    y = n_pedestal * bin_width * asymmetric_gaussian_pdf(x, pedestal, sigma_ped, sigma_ped)

    for n_pe, n_peak in enumerate(peak_areas, start=1):
        mean = pedestal + n_pe * spe_area
        sigma_left = np.sqrt(sigma_ped**2 + n_pe * sigma_spe_left**2)
        sigma_right = np.sqrt(sigma_ped**2 + n_pe * sigma_spe_right**2)
        y += n_peak * bin_width * asymmetric_gaussian_pdf(x, mean, sigma_left, sigma_right)

    return y + constant_per_bin


def free_spe_components(x, parameters, bin_width, max_pe: int = 3):
    """Return the pedestal, PE, and constant components of ``free_spe_model``."""
    pedestal = parameters["q0_mV_ns"]
    sigma_ped = parameters["sigma0_mV_ns"]
    spe_area = parameters["q1_mV_ns"]
    sigma_left = parameters["sigma1_left_mV_ns"]
    sigma_right = parameters["sigma1_right_mV_ns"]

    components = {
        "pedestal": parameters["n_pedestal"]
        * bin_width
        * asymmetric_gaussian_pdf(x, pedestal, sigma_ped, sigma_ped),
        "constant": np.full_like(x, parameters["constant_per_bin"], dtype=float),
    }

    for n_pe in range(1, max_pe + 1):
        mean = pedestal + n_pe * spe_area
        peak_sigma_left = np.sqrt(sigma_ped**2 + n_pe * sigma_left**2)
        peak_sigma_right = np.sqrt(sigma_ped**2 + n_pe * sigma_right**2)
        components[f"{n_pe} PE"] = (
            parameters[f"n_{n_pe}pe"]
            * bin_width
            * asymmetric_gaussian_pdf(x, mean, peak_sigma_left, peak_sigma_right)
        )

    return components


def fit_free_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 3,
    p0=None,
    maxfev: int = 100000,
):
    """Fit a charge spectrum with free PE peak areas and asymmetric SPE widths.

    ``p0`` must be ``{param_name: [initial, lower, upper, is_fixed]}``.
    """
 
    fit = fit_histogram_model(
        charge_mV_ns,
        free_spe_model,
        free_spe_parameter_names(max_pe=max_pe),
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_name="free_spe",
        model_kwargs={"max_pe": max_pe},
        maxfev=maxfev,
    )
    fit["max_pe"] = max_pe
    return fit


# --------- Bellamy ---------


def bellamy_spe_parameter_names() -> list[str]:
    return [
        "n_total",
        "mu_pe",
        "q0_mV_ns",
        "sigma0_mV_ns",
        "q1_mV_ns",
        "sigma1_mV_ns",
        "background_probability",
        "background_slope_per_mV_ns",
    ]


def bellamy_spe_model(
    x,
    n_total,
    mu_pe,
    pedestal,
    sigma_ped,
    spe_area,
    sigma_spe,
    background_probability,
    background_slope,
    bin_width,
    max_pe: int = 8,
):
    """Bellamy et al. PMT response model.

    This is the Poisson-weighted PM response where each n-PE Gaussian response is
    mixed with an exponential-background convolution. ``background_probability``
    is the paper's w parameter and ``background_slope`` is the exponential
    coefficient a.
    """
    y = np.zeros_like(x, dtype=float)
    mu_pe = np.maximum(mu_pe, 1e-12)
    w = np.clip(background_probability, 0.0, 1.0)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = pedestal + n_pe * spe_area
        sigma = np.sqrt(sigma_ped**2 + n_pe * sigma_spe**2)
        response = (1.0 - w) * gaussian_pdf(x, mean, sigma)
        response += w * exponential_gaussian_pdf(x, mean, sigma, background_slope)
        y += n_total * bin_width * weight * response

    return y


def bellamy_spe_components(x, parameters, bin_width, max_pe: int = 8):
    """Return Bellamy model components grouped by n photoelectrons."""
    components = {}
    p = parameters
    mu_pe = np.maximum(p["mu_pe"], 1e-12)
    w = np.clip(p["background_probability"], 0.0, 1.0)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = p["q0_mV_ns"] + n_pe * p["q1_mV_ns"]
        sigma = np.sqrt(p["sigma0_mV_ns"] ** 2 + n_pe * p["sigma1_mV_ns"] ** 2)
        response = (1.0 - w) * gaussian_pdf(x, mean, sigma)
        response += w * exponential_gaussian_pdf(x, mean, sigma, p["background_slope_per_mV_ns"])
        components[n_pe] = p["n_total"] * bin_width * weight * response

    return components


def fit_bellamy_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 8,
    p0=None,
    maxfev: int = 100000,
):
    """Fit the charge spectrum with the Bellamy et al. PMT response model.

    ``p0`` must be ``{param_name: [initial, lower, upper, is_fixed]}``.
    """
    fit = fit_histogram_model(
        charge_mV_ns,
        bellamy_spe_model,
        bellamy_spe_parameter_names(),
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_name="bellamy_spe",
        model_kwargs={"max_pe": max_pe},
        maxfev=maxfev,
    )
    fit["max_pe"] = max_pe
    return fit

# --------- 2 paths ---------

def dynode_spe_parameter_names() -> list[str]:
    return [
        "n_total",
        "mu_pe",
        "q0_mV_ns",
        "sigma0_mV_ns",
        "g1_mV_ns",
        "sigma_g1_mV_ns",
        "g2_mV_ns",
        "sigma_g2_mV_ns",
        "alpha",
    ]


def dynode_spe_model(
    x,
    n_total,
    mu_pe,
    pedestal,
    sigma_ped,
    g1,
    sigma_g1,
    g2,
    sigma_g2,
    alpha,
    bin_width,
    max_pe: int = 8,
):
    """Poisson PE model with binomial first/second-dynode amplification paths.

    For ``n`` photoelectrons, ``k`` are assigned to the second-dynode path with
    binomial probability ``Binomial(k | n, alpha)``. The remaining ``n-k`` use
    the first-dynode path. Each branch contributes a Gaussian charge response.
    """
    y = np.zeros_like(x, dtype=float)
    mu_pe = np.maximum(mu_pe, 1e-12)
    alpha = np.clip(alpha, 0.0, 1.0)

    for n_pe in range(max_pe + 1):
        log_poisson = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        poisson_weight = np.exp(log_poisson)

        for n_second in range(n_pe + 1):
            n_first = n_pe - n_second
            if alpha == 0.0 and n_second > 0:
                continue
            if alpha == 1.0 and n_first > 0:
                continue

            log_binomial = (
                gammaln(n_pe + 1)
                - gammaln(n_second + 1)
                - gammaln(n_first + 1)
            )
            if n_second:
                log_binomial += n_second * np.log(np.maximum(alpha, 1e-300))
            if n_first:
                log_binomial += n_first * np.log(np.maximum(1.0 - alpha, 1e-300))

            weight = poisson_weight * np.exp(log_binomial)
            mean = pedestal + n_first * g1 + n_second * g2
            sigma = np.sqrt(sigma_ped**2 + n_first * sigma_g1**2 + n_second * sigma_g2**2)
            sigma = np.maximum(sigma, 1e-9)
            y += n_total * bin_width * weight / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                -0.5 * ((x - mean) / sigma) ** 2
            )

    return y


def dynode_spe_components(x, parameters, bin_width, max_pe: int = 8):
    """Return dynode-path model components grouped by ``n`` photoelectrons."""
    components = {}
    for n_pe in range(max_pe + 1):
        component = np.zeros_like(x, dtype=float)
        for n_second in range(n_pe + 1):
            p = dict(parameters)
            # One state at a time: total normalization times the exact state
            # probability. Reuse the same Gaussian expression as the model.
            n_first = n_pe - n_second
            alpha = np.clip(p["alpha"], 0.0, 1.0)
            log_poisson = -p["mu_pe"] + n_pe * np.log(np.maximum(p["mu_pe"], 1e-12)) - gammaln(n_pe + 1)
            log_binomial = (
                gammaln(n_pe + 1)
                - gammaln(n_second + 1)
                - gammaln(n_first + 1)
            )
            if n_second:
                log_binomial += n_second * np.log(np.maximum(alpha, 1e-300))
            if n_first:
                log_binomial += n_first * np.log(np.maximum(1.0 - alpha, 1e-300))
            weight = np.exp(log_poisson + log_binomial)
            mean = p["q0_mV_ns"] + n_first * p["g1_mV_ns"] + n_second * p["g2_mV_ns"]
            sigma = np.sqrt(
                p["sigma0_mV_ns"] ** 2
                + n_first * p["sigma_g1_mV_ns"] ** 2
                + n_second * p["sigma_g2_mV_ns"] ** 2
            )
            sigma = np.maximum(sigma, 1e-9)
            component += p["n_total"] * bin_width * weight / (np.sqrt(2 * np.pi) * sigma) * np.exp(
                -0.5 * ((x - mean) / sigma) ** 2
            )
        components[n_pe] = component
    return components


def fit_dynode_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 8,
    p0=None,
    maxfev: int = 100000,
):
    """Fit the charge spectrum with the ProtoDUNE dynode-path SPE model.

    ``p0`` must be ``{param_name: [initial, lower, upper, is_fixed]}``.
    """
    fit = fit_histogram_model(
        charge_mV_ns,
        dynode_spe_model,
        dynode_spe_parameter_names(),
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_name="dynode_spe",
        model_kwargs={"max_pe": max_pe},
        maxfev=maxfev,
    )
    fit["max_pe"] = max_pe
    return fit


# --------- Fit Helpers and wrappers ---------

def parse_fit_parameter_specs(p0, parameter_names):
    """Parse ``{name: [initial, lower, upper, is_fixed]}`` fit specs.

    Fixed parameters are included in the model parameter dictionary but are not
    varied by ``curve_fit``.
    """
    if p0 is None:
        raise ValueError(
            "p0 must be provided as {param_name: [initial, lower_bound, upper_bound, is_fixed]}"
        )

    missing = [name for name in parameter_names if name not in p0]
    if missing:
        raise KeyError(f"Missing p0 entries for: {', '.join(missing)}")

    unknown = [name for name in p0 if name not in parameter_names]
    if unknown:
        raise KeyError(f"Unknown p0 entries for this model: {', '.join(unknown)}")

    initial_parameters = {}
    bounds_lower = {}
    bounds_upper = {}
    fixed_parameters = {}
    fitted_parameter_names = []
    fitted_initial = []
    fitted_lower = []
    fitted_upper = []

    for name in parameter_names:
        spec = p0[name]
        if isinstance(spec, dict):
            initial = spec["initial"]
            lower = spec["lower"]
            upper = spec["upper"]
            is_fixed = spec.get("is_fixed", spec.get("fixed", False))
        else:
            values = list(spec)
            if len(values) != 4:
                raise ValueError(
                    f"p0[{name!r}] must be [initial, lower_bound, upper_bound, is_fixed]"
                )
            initial, lower, upper, is_fixed = values

        initial = float(initial)
        lower = float(lower)
        upper = float(upper)
        is_fixed = bool(is_fixed)

        if lower > upper:
            raise ValueError(f"Lower bound is above upper bound for {name!r}")
        if is_fixed:
            lower = initial
            upper = initial
        elif not (lower <= initial <= upper):
            raise ValueError(
                f"Initial value for {name!r} is outside its bounds: {initial} not in [{lower}, {upper}]"
            )

        initial_parameters[name] = initial
        bounds_lower[name] = lower
        bounds_upper[name] = upper

        if is_fixed:
            fixed_parameters[name] = initial
        else:
            fitted_parameter_names.append(name)
            fitted_initial.append(initial)
            fitted_lower.append(lower)
            fitted_upper.append(upper)

    return {
        "initial_parameters": initial_parameters,
        "bounds": {
            "lower": bounds_lower,
            "upper": bounds_upper,
        },
        "fixed_parameters": fixed_parameters,
        "fitted_parameter_names": fitted_parameter_names,
        "fitted_initial": np.asarray(fitted_initial, dtype=float),
        "fitted_bounds": (
            np.asarray(fitted_lower, dtype=float),
            np.asarray(fitted_upper, dtype=float),
        ),
    }


def fit_histogram_model(
    charge_mV_ns,
    model_function,
    parameter_names,
    p0,
    fit_range=None,
    bins: int = 250,
    model_name: str = "custom",
    model_kwargs=None,
    maxfev: int = 100000,
):
    """Fit a histogram with an arbitrary model and fixed/free parameters.

    ``model_function`` must accept ``model_function(x, *params, **model_kwargs)``,
    where ``params`` follow ``parameter_names`` order.
    """
    charge_mV_ns = np.asarray(charge_mV_ns, dtype=float)
    charge_mV_ns = charge_mV_ns[np.isfinite(charge_mV_ns)]
    if fit_range is None:
        fit_range = default_fit_range(charge_mV_ns)
    if model_kwargs is None:
        model_kwargs = {}

    counts, edges = np.histogram(charge_mV_ns, bins=bins, range=fit_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    specs = parse_fit_parameter_specs(p0, parameter_names)
    initial_parameters = specs["initial_parameters"]
    fitted_parameter_names = specs["fitted_parameter_names"]

    def full_parameters(fitted_values):
        parameters = dict(initial_parameters)
        parameters.update(zip(fitted_parameter_names, fitted_values))
        return parameters

    def model_from_dict(x, parameters):
        ordered = [parameters[name] for name in parameter_names]
        return model_function(x, *ordered, bin_width=bin_width, **model_kwargs)

    sigma_counts = np.sqrt(np.maximum(counts, 1))

    if fitted_parameter_names:
        def model_for_fit(x, *fitted_values):
            return model_from_dict(x, full_parameters(fitted_values))

        popt_free, pcov_free = curve_fit(
            model_for_fit,
            centers,
            counts,
            p0=specs["fitted_initial"],
            bounds=specs["fitted_bounds"],
            sigma=sigma_counts,
            absolute_sigma=True,
            maxfev=maxfev,
        )
        parameters = full_parameters(popt_free)
    else:
        popt_free = np.array([], dtype=float)
        pcov_free = np.empty((0, 0), dtype=float)
        parameters = dict(initial_parameters)

    model_counts = model_from_dict(centers, parameters)
    chi2 = np.sum(((counts - model_counts) / sigma_counts) ** 2)
    ndof = len(counts) - len(fitted_parameter_names)

    errors = {name: 0.0 if name in specs["fixed_parameters"] else np.nan for name in parameter_names}
    for i, name in enumerate(fitted_parameter_names):
        errors[name] = float(np.sqrt(pcov_free[i, i])) if pcov_free.size else np.nan

    diagnostics = []
    lower = specs["bounds"]["lower"]
    upper = specs["bounds"]["upper"]
    for name in fitted_parameter_names:
        value = parameters[name]
        low = lower[name]
        high = upper[name]
        if np.isclose(value, low, rtol=0, atol=max(1e-6, 1e-3 * max(abs(low), 1.0))):
            diagnostics.append(f"{name} is close to the lower fit bound")
        if np.isfinite(high) and np.isclose(value, high, rtol=0, atol=max(1e-6, 1e-3 * max(abs(high), 1.0))):
            diagnostics.append(f"{name} is close to the upper fit bound")

    return {
        "model": model_name,
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "bin_width": bin_width,
        "fit_range": fit_range,
        "parameter_names": list(parameter_names),
        "fitted_parameter_names": fitted_parameter_names,
        "fixed_parameters": specs["fixed_parameters"],
        "parameters": parameters,
        "errors": errors,
        "covariance": pcov_free,
        "model_counts": model_counts,
        "chi2": float(chi2),
        "ndof": int(ndof),
        "diagnostics": diagnostics,
        "initial_parameters": initial_parameters,
        "bounds": specs["bounds"],
        "model_kwargs": dict(model_kwargs),
    }


def default_fit_range(charge_mV_ns, low_quantile=0.001, high_quantile=0.98, pad_fraction=0.05):
    low, high = np.quantile(charge_mV_ns, [low_quantile, high_quantile])
    pad = pad_fraction * (high - low)
    return float(low - pad), float(high + pad)



def fit_result_table(fit, as_dataframe=True):
    """Summarize initial values, bounds, fixed flags, values, and errors.

    Parameters
    ----------
    fit
        Fit dictionary returned by ``fit_histogram_model`` or one of its model
        wrappers.
    as_dataframe
        If True, return a pandas DataFrame when pandas is available. If pandas
        is not installed, a list of dictionaries is returned instead.
    """
    parameter_names = fit.get("parameter_names")
    if parameter_names is None:
        parameter_names = list(fit.get("parameters", {}).keys())

    initial = fit.get("initial_parameters", {})
    bounds = fit.get("bounds", {})
    lower = bounds.get("lower", {})
    upper = bounds.get("upper", {})
    fixed_parameters = fit.get("fixed_parameters", {})
    parameters = fit.get("parameters", {})
    errors = fit.get("errors", {})

    rows = []
    for name in parameter_names:
        rows.append(
            {
                "parameter": name,
                "initial": initial.get(name, np.nan),
                "lower_bound": lower.get(name, np.nan),
                "upper_bound": upper.get(name, np.nan),
                "fixed": name in fixed_parameters,
                "fit_value": parameters.get(name, np.nan),
                "fit_error": errors.get(name, np.nan),
            }
        )

    if as_dataframe:
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            pass
    return rows

# ----------------------------------------------------------------------------------------------
# Plots / Printouts
# ----------------------------------------------------------------------------------------------

def plot_average_waveforms(time_ns, average_results, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    for result in average_results:
        wf = result["average_waveform"]
        if wf is None:
            continue

        low, high = result["bin"]
        ax.plot(time_ns, wf, label=f"{low}-{high} mV (N={result['n_events']})")

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def fit_component_function(fit):
    """Return the component helper corresponding to a standard fit model."""
    model = fit.get("model")
    if model == "poisson_spe":
        return poisson_spe_components
    if model == "bellamy_spe":
        return bellamy_spe_components
    if model == "dynode_spe":
        return dynode_spe_components
    if model == "free_spe":
        return free_spe_components
    return None


def plot_fit_result(
    fit,
    title=None,
    component_function=None,
    component_kwargs=None,
    ax=None,
    ax_resid=None,
    show_components=True,
    show_errorbars=True,
    logscale=True,
    ylims=None,
    residual_ylim=(-8, 8),
):
    """Plot histogram data, total fit, model components, and residuals.

    Parameters
    ----------
    fit
        Fit dictionary returned by ``fit_histogram_model`` or one of its model
        wrappers.
    component_function
        Optional callable with signature ``f(x, parameters, bin_width, **kwargs)``.
        If omitted, the function is chosen automatically for known models.
    component_kwargs
        Extra keyword arguments passed to ``component_function``. By default,
        ``max_pe`` is taken from the fit dictionary when available.
    """

    if ax is None or ax_resid is None:
        fig, (ax, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(9, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )
    else:
        fig = ax.figure

    centers = fit["centers"]
    counts = fit["counts"]
    model_counts = fit["model_counts"]
    counts_unc = np.sqrt(np.maximum(counts, 1))
    residuals = (counts - model_counts) / counts_unc

    if show_errorbars:
        ax.errorbar(
            centers,
            counts,
            yerr=counts_unc,
            fmt=".",
            ms=3,
            color="black",
            alpha=0.75,
            label="data",
        )
    else:
        ax.plot(centers, counts, ".", ms=3, color="black", alpha=0.75, label="data")

    ax.plot(centers, model_counts, color="tab:red", lw=2.2, label="total fit")

    if show_components:
        if component_function is None:
            component_function = fit_component_function(fit)
        if component_function is not None:
            if component_kwargs is None:
                component_kwargs = {}
            if "max_pe" not in component_kwargs and "max_pe" in fit:
                component_kwargs = {**component_kwargs, "max_pe": fit["max_pe"]}
            components = component_function(
                centers,
                fit["parameters"],
                fit["bin_width"],
                **component_kwargs,
            )
            for key, y_component in components.items():
                label = f"{key} PE" if isinstance(key, (int, np.integer)) else str(key)
                ax.plot(centers, y_component, ls="--", lw=1.4, label=label)

    if title is None:
        model = fit.get("model", "fit")
        chi2_ndof = fit["chi2"] / fit["ndof"] if fit["ndof"] else np.nan
        title = f"{model} (chi2/ndof = {chi2_ndof:.3g})"
    ax.set_title(title)
    ax.set_ylabel("Events / bin")
    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.legend(ncol=2, fontsize=8)

    ax_resid.axhline(0, color="black", lw=1)
    ax_resid.axhline(2, color="0.55", lw=0.8, ls=":")
    ax_resid.axhline(-2, color="0.55", lw=0.8, ls=":")
    if show_errorbars:
        ax_resid.errorbar(
            centers,
            residuals,
            yerr=np.ones_like(residuals),
            fmt=".",
            color="tab:gray",
            ms=3,
            elinewidth=0.8,
            capsize=0,
        )
    else:
        ax_resid.plot(centers, residuals, ".", color="tab:gray", ms=3)
    ax_resid.set_xlabel("Charge [mV ns]")
    ax_resid.set_ylabel("Residual\n[data-fit]/sigma")
    if residual_ylim is not None:
        ax_resid.set_ylim(residual_ylim)

    return fig, ax, ax_resid


def print_fit_result_table(fit):
    """Print the compact fit-result table and fit-quality diagnostics."""
    table = fit_result_table(fit, as_dataframe=True)
    if hasattr(table, "to_string"):
        print(table.to_string(index=False))
    else:
        print(
            f"{'parameter':<25} {'initial':>15} {'lower_bound':>15} "
            f"{'upper_bound':>15} {'fixed':>8} {'fit_value':>15} {'fit_error':>15}"
        )
        print("-" * 115)
        for row in table:
            print(
                f"{row['parameter']:<25} "
                f"{row['initial']:>15.6g} "
                f"{row['lower_bound']:>15.6g} "
                f"{row['upper_bound']:>15.6g} "
                f"{str(row['fixed']):>8} "
                f"{row['fit_value']:>15.6g} "
                f"{row['fit_error']:>15.6g}"
            )

    chi2 = fit.get("chi2", np.nan)
    ndof = fit.get("ndof", 0)
    if ndof:
        print(f"\nchi2 / ndof = {chi2:.1f} / {ndof} = {chi2 / ndof:.3f}")
    else:
        print(f"\nchi2 / ndof = {chi2:.1f} / {ndof}")

    if fit.get("diagnostics"):
        print("\nDiagnostics:")
        for diagnostic in fit["diagnostics"]:
            print(f"  - {diagnostic}")


def print_spe_fit_result(fit):
    if "initial_parameters" in fit and "bounds" in fit:
        print("\nInitial fit parameters and bounds:")
        print("-" * 95)
        print(f"{'Parameter':<25} {'Initial':>15} {'Lower bound':>20} {'Upper bound':>20} {'Fixed':>10}")
        print("-" * 95)

        for name in fit["initial_parameters"]:
            val = fit["initial_parameters"][name]
            low = fit["bounds"]["lower"][name]
            high = fit["bounds"]["upper"][name]
            low_str = f"{low:.6g}" if np.isfinite(low) else "-inf"
            high_str = f"{high:.6g}" if np.isfinite(high) else "inf"
            fixed = name in fit.get("fixed_parameters", {})
            print(f"{name:<25} {val:>15.6g} {low_str:>20} {high_str:>20} {str(fixed):>10}")

        print("-" * 95)

    print("\nFit results:")
    print("-" * 70)
    print(f"{'Parameter':<25} {'Value':>15} {'Error':>15}")
    print("-" * 70)

    for name in fit["parameters"]:
        val = fit["parameters"][name]
        err = fit["errors"].get(name, np.nan)

        print(
            f"{name:<25} "
            f"{val:>15.6g} "
            f"{err:>15.6g}"
        )

    print("-" * 70)

    chi2 = fit["chi2"]
    ndof = fit["ndof"]

    print(
        f"chi2 / ndof = {chi2:.1f} / {ndof} = {chi2 / ndof:.3f}"
    )

    if fit["diagnostics"]:
        print("\nDiagnostics:")
        for diagnostic in fit["diagnostics"]:
            print(f"  - {diagnostic}")

    fixed_parameters = fit.get("fixed_parameters", {})
    if fixed_parameters:
        print("\nFixed parameters:")
        for name, value in fixed_parameters.items():
            print(f"  - {name} = {value:.6g}")

