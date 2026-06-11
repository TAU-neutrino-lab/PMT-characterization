from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pmt_analysis import read_keysight_h5_direct, standard_units

# ------------------------------------------------------------------
# File info:
# files are 1s continuous trace (no segments)
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Baseline Removal
# ------------------------------------------------------------------
"""
 Data is mostly quiet except few pulses: use full trace
    compute median (less affected by outliers than mean)
    subtract median from raw data

 noise:
    use median absolute deviation (MAD)
    MAD = median( abs(baseline_subtracted_trace) )
    convert MAD into Gaussin-equivalent noise sigma (MAD = 0.67449 * sigma => sigma = MAD/0.67449 = 1.4826 * MAD)
"""

def robust_location_scale(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0:
        sigma = np.std(values)
    return median, sigma


def baseline_subtract_trace(voltage_mV):
    """main function"""
    baseline_mV, noise_mV = robust_location_scale(voltage_mV)
    return voltage_mV - baseline_mV, baseline_mV, noise_mV

# ------------------------------------------------------------------
# Pulse Finding
# ------------------------------------------------------------------
"""
 1. make negative pulses positive (easier for peak_finding algorithm)
 2. look for regions above threshold_mV (contiguous regions where the signal stays above threshold) 
 3. merge close regions: if two above-threshold reigons are closer than min_separation_samples they are merged into one candidate
 4. in each region, find the max (i.e. the min for negative pulses)
 5. measure timing shape: 
    - width50_ns: width between leading/trailing crossings at 50% of peak amplitude
    - rise_time_ns: leading-edge time from 10% to 90% of peak amplitude
6. measure charge: integrates the baseline-subtracted waveform around the peak
"""

def threshold_regions(signal_mV, threshold_mV):
    above = signal_mV >= threshold_mV
    edges = np.diff(above.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, len(above)]
    return starts, stops

def merge_close_regions(starts, stops, min_separation_samples):
    if len(starts) == 0:
        return starts, stops
    merged_starts = [int(starts[0])]
    merged_stops = [int(stops[0])]
    for start, stop in zip(starts[1:], stops[1:]):
        if int(start) - merged_stops[-1] <= min_separation_samples:
            merged_stops[-1] = int(stop)
        else:
            merged_starts.append(int(start))
            merged_stops.append(int(stop))
    return np.array(merged_starts, dtype=int), np.array(merged_stops, dtype=int)

def leading_edge_crossing_ns(time_ns, signal, threshold, peak_idx, left_limit):
    idx = int(peak_idx)
    while idx > left_limit and signal[idx] >= threshold:
        idx -= 1
    if idx == peak_idx:
        return float(time_ns[peak_idx])
    if signal[idx] >= threshold:
        return float(time_ns[left_limit])
    x0, x1 = time_ns[idx], time_ns[idx + 1]
    y0, y1 = signal[idx], signal[idx + 1]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def trailing_edge_crossing_ns(time_ns, signal, threshold, peak_idx, right_limit):
    idx = int(peak_idx)
    last = min(int(right_limit), len(signal) - 1)
    while idx < last and signal[idx] >= threshold:
        idx += 1
    if idx == peak_idx:
        return float(time_ns[peak_idx])
    if signal[idx] >= threshold:
        return float(time_ns[last])
    x0, x1 = time_ns[idx - 1], time_ns[idx]
    y0, y1 = signal[idx - 1], signal[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))

def measure_pulses(time_ns, voltage_bs_mV, sample_dt_ns, config_dict):
    signal = -voltage_bs_mV
    min_separation_samples = max(1, int(round(config_dict['min_event_separation_ns'] / sample_dt_ns)))
    charge_pre = int(round(config_dict['charge_pre_ns'] / sample_dt_ns))
    charge_post = int(round(config_dict['charge_post_ns'] / sample_dt_ns))
    edge_pre = int(round(config_dict['pre_event_ns'] / sample_dt_ns))
    edge_post = int(round(config_dict['post_event_ns'] / sample_dt_ns))

    starts, stops = threshold_regions(signal, config_dict['event_threshold_mV'])
    starts, stops = merge_close_regions(starts, stops, min_separation_samples)

    rows = []
    for start, stop in zip(starts, stops):
        local_peak = int(start + np.argmax(signal[start:stop]))
        amp = float(signal[local_peak])
        half = 0.5 * amp
        low = config_dict['rise_low_fracion'] * amp
        high = config_dict['rise_high_fracion'] * amp
        left_limit = max(0, local_peak - edge_pre)
        right_limit = min(len(signal) - 1, local_peak + edge_post)

        t_half_left = leading_edge_crossing_ns(time_ns, signal, half, local_peak, left_limit)
        t_half_right = trailing_edge_crossing_ns(time_ns, signal, half, local_peak, right_limit)
        width50 = t_half_right - t_half_left

        t_low = leading_edge_crossing_ns(time_ns, signal, low, local_peak, left_limit)
        t_high = leading_edge_crossing_ns(time_ns, signal, high, local_peak, left_limit)
        rise_time = t_high - t_low if t_high >= t_low else np.nan

        q0 = max(0, local_peak - charge_pre)
        q1 = min(len(time_ns), local_peak + charge_post + 1)
        charge = -np.trapezoid(voltage_bs_mV[q0:q1], x=time_ns[q0:q1])

        rows.append({
            "start_index": int(start),
            "stop_index": int(stop),
            "peak_index": int(local_peak),
            "peak_time_ns": float(time_ns[local_peak]),
            "peak_time_s": float((time_ns[local_peak] - time_ns[0]) * 1e-9),
            "peak_amplitude_mV": amp,
            "charge_mV_ns": float(charge),
            "width50_ns": float(width50),
            "rise_time_ns": float(rise_time),
            "positive_overshoot_mV": float(np.max(voltage_bs_mV[q0:q1])),
        })
    return rows

# ------------------------------------------------------------------
# Process all files
# ------------------------------------------------------------------

def read_continuous_trace(file):
    time_s, voltage_V, metadata = read_keysight_h5_direct(file)
    time_ns, voltage_mV, adc_step_mV, *_ = standard_units(time_s, voltage_V, metadata)
    if voltage_mV.shape[0] != 1:
        raise ValueError(f"Expected one continuous trace in {file}, got shape {voltage_mV.shape}")
    return time_ns, voltage_mV[0], metadata, adc_step_mV

def rows_to_arrays(rows):
    # convert lists of dicts into dicts of np arrays (easier for analysis)
    if not rows:
        return {}
    keys = rows[0].keys()
    arrays = {}
    for key in keys:
        values = [row[key] for row in rows]
        if isinstance(values[0], str):
            arrays[key] = np.array(values, dtype=object)
        else:
            arrays[key] = np.array(values)
    return arrays

def process_all_files(files, sample_dt_ns, config_dict):
    all_rows = []
    file_summaries = []
    reference_time_ns = None

    for file_index, file in enumerate(files):
        t_ns, v_mV, meta, adc = read_continuous_trace(file)
        if reference_time_ns is None:
            reference_time_ns = t_ns
        elif len(t_ns) != len(reference_time_ns) or not np.allclose(np.diff(t_ns[:10]), np.diff(reference_time_ns[:10])):
            print(f"Warning: time axis differs in {file}")

        v_bs_mV, file_baseline_mV, file_noise_mV = baseline_subtract_trace(v_mV)
        rows = measure_pulses(t_ns, v_bs_mV, sample_dt_ns, config_dict)
        for row in rows:
            row["file"] = str(file)
            row["file_index"] = file_index
            row["baseline_mV"] = file_baseline_mV
            row["noise_mV"] = file_noise_mV
        all_rows.extend(rows)

        duration_s = len(t_ns) * float(np.median(np.diff(t_ns))) * 1e-9
        file_summaries.append({
            "file": str(file),
            "file_index": file_index,
            "duration_s": duration_s,
            "n_events": len(rows),
            "rate_hz": len(rows) / duration_s,
            "baseline_mV": file_baseline_mV,
            "noise_mV": file_noise_mV,
            "layout": meta.get("layout"),
        })
    events = rows_to_arrays(all_rows)
    file_summary = rows_to_arrays(file_summaries)
    return events, file_summary

def get_mean_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.median(values))

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

# def add_mean_median_text(ax, values, unit="", loc=(0.98, 0.94)):
#     mean, median = get_mean_median(values)
#     unit_text = f" {unit}" if unit else ""
#     ax.text(
#         loc[0],
#         loc[1],
#         f"mean = {mean:.4g}{unit_text} median = {median:.4g}{unit_text}",
#         transform=ax.transAxes,
#         ha="right",
#         va="top",
#         fontsize=9,
#         bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.9},
#     )
#     return mean, median

def plot_event_variables_per_file(
    events,
    files,
    plot_specs,
    bins=80,
    ncols=3,
    same_plot=False,
    histtype="step",
):
    """
    Plot event-variable histograms split by file.

    Parameters
    ----------
    events : dict of np.ndarray
        Output of rows_to_arrays(all_rows).
    files : list[Path]
        File list, used for labels.
    plot_specs : list[tuple]
        Tuples of (name, xlabel, unit, log_y, hist_range).
    bins : int
        Histogram bins.
    ncols : int
        Number of subplot columns when same_plot=False.
    same_plot : bool
        If False: one figure per variable, with one subplot per file.
        If True: one figure per variable, all files overlaid on one axis.
    """

    if not events:
        print("No events to plot.")
        return

    n_files = len(files)

    for name, xlabel, unit, log_y, hist_range in plot_specs:
        if same_plot:
            fig, ax = plt.subplots(figsize=(8, 5))

            for file_index, file in enumerate(files):
                mask = events["file_index"] == file_index
                values = events[name][mask]
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue

                mean, median = get_mean_median(values)

                ax.hist(
                    values,
                    bins=bins,
                    histtype=histtype,
                    lw=1.5,
                    label=(
                        f"{file.name}: "
                        f"mean={mean:.3g} {unit}, "
                        f"median={median:.3g} {unit}"
                    ),
                    range=hist_range
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Events")
            ax.set_title(name)
            if log_y:
                ax.set_yscale("log")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()

        else:
            nrows = int(np.ceil(n_files / ncols))

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5 * ncols, 3.4 * nrows),
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_1d(axes).ravel()

            for file_index, (ax, file) in enumerate(zip(axes, files)):
                mask = events["file_index"] == file_index
                values = events[name][mask]
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    ax.set_title(f"{file.name}\nno events")
                    continue

                mean, median = get_mean_median(values)

                ax.hist(values, bins=bins, histtype=histtype, color="tab:blue", range=hist_range)
                ax.axvline(
                    mean,
                    color="tab:orange",
                    ls=":",
                    lw=1.8,
                    label=f"mean = {mean:.3g} {unit}",
                )
                ax.axvline(
                    median,
                    color="black",
                    ls="--",
                    lw=1.2,
                    label=f"median = {median:.3g} {unit}",
                )

                ax.set_title(file.name)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Events")
                if log_y:
                    ax.set_yscale("log")
                ax.legend(loc="best", fontsize=8)

            for ax in axes[n_files:]:
                ax.set_axis_off()

            fig.suptitle(name, y=1.02)
            fig.tight_layout()