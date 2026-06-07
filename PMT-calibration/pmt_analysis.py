"""Reusable PMT LED calibration helpers.

The charge convention used here is:
    charge_mV_ns = - integral(baseline-subtracted voltage_mV dt_ns)

This makes negative-going PMT pulses have positive charge, with units mV ns.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln


SEGMENT_RE = re.compile(r"Seg(\d+)Data$")


def segment_number(name: str) -> int | None:
    match = SEGMENT_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def sorted_segment_keys(channel_group) -> list[str]:
    keys = [key for key in channel_group.keys() if segment_number(key) is not None]
    return sorted(keys, key=segment_number)


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
    """Read a Keysight segmented HDF5 file into voltage in V.

    This replaces the lab-specific ``lab_tools.io.read_keysight_h5`` helper and
    returns ``time_s, voltage_V, metadata``.
    """
    filename = Path(filename)
    with h5py.File(filename, "r") as h5_file:
        channel_group = keysight_channel_group(h5_file, channel=channel)
        attrs = channel_group.attrs
        keys = sorted_segment_keys(channel_group)

        if segment_numbers is not None:
            wanted = set(segment_numbers)
            keys = [key for key in keys if segment_number(key) in wanted]

        time_s = keysight_time_axis_ns(channel_group) * 1e-9
        yinc = float(attrs["YInc"])
        yorg = float(attrs["YOrg"])

        voltage = np.empty((len(keys), len(time_s)), dtype=dtype)
        for i, key in enumerate(keys):
            voltage[i] = channel_group[key][()] * yinc + yorg

        metadata = {
            "filename": str(filename),
            "channel": channel,
            "channel_attrs": dict(attrs.items()),
            "segment_numbers": [segment_number(key) for key in keys],
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
                    },
                }


def standard_units(time, voltage, metadata):
    """Convert a direct-reader output from seconds/volts to ns/mV."""
    voltage_mV = voltage * 1e3
    adc_step_mV = metadata["channel_attrs"]["YInc"] * 1e3
    time_ns = time * 1e9
    return time_ns, voltage_mV, adc_step_mV, "ns", "mV"


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


def compute_charge_around_minimum(time_ns, voltage_bs_mV, pre_samples: int = 20, post_samples: int = 40):
    """Integrate around each waveform minimum and return positive charge for negative pulses."""
    idx_min = np.argmin(voltage_bs_mV, axis=1)
    charge = np.zeros(len(voltage_bs_mV), dtype=np.float64)

    for i, idx in enumerate(idx_min):
        start = max(0, idx - pre_samples)
        stop = min(len(time_ns), idx + post_samples)
        charge[i] = -np.trapezoid(voltage_bs_mV[i, start:stop], x=time_ns[start:stop])

    return charge


def summarize_waveforms(files, baseline_window_ns, channel: str = "Channel 1", chunk_size: int = 512):
    """Compute baseline QA quantities and the average baseline-subtracted waveform."""
    n_events = 0
    mean_waveform = None
    m2_waveform = None
    baseline_values = []
    baseline_check_values = []
    pulse_min_times = []
    pulse_amplitudes = []
    time_ns = None

    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        time_ns = chunk["time_ns"]
        voltage_bs_mV, baseline_mV = subtract_baseline(
            time_ns,
            chunk["voltage_mV"],
            baseline_window_ns=baseline_window_ns,
        )

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


def compute_charges_for_window(
    files,
    baseline_window_ns,
    integration_window_ns,
    channel: str = "Channel 1",
    chunk_size: int = 512,
):
    charges = []
    for chunk in iter_keysight_chunks(files, channel=channel, chunk_size=chunk_size):
        voltage_bs_mV, _ = subtract_baseline(
            chunk["time_ns"],
            chunk["voltage_mV"],
            baseline_window_ns=baseline_window_ns,
        )
        charges.append(compute_charge_window(chunk["time_ns"], voltage_bs_mV, integration_window_ns))
    return np.concatenate(charges)


def compute_charges_for_windows(
    files,
    baseline_window_ns,
    integration_windows_ns,
    channel: str = "Channel 1",
    chunk_size: int = 512,
):
    """Compute charges for many fixed windows in one pass through the files."""
    windows = [tuple(map(float, window)) for window in integration_windows_ns]
    charge_parts = {window: [] for window in windows}

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

        for window in windows:
            start = int(np.searchsorted(time_ns, window[0], side="left"))
            stop = int(np.searchsorted(time_ns, window[1], side="right") - 1)
            if stop <= start:
                raise ValueError(f"Integration window {window} contains too few samples")
            charge_parts[window].append(-(cumulative[:, stop] - cumulative[:, start]))

    return {window: np.concatenate(parts) for window, parts in charge_parts.items()}


def gaussian(x, amplitude, mean, sigma):
    sigma = np.maximum(sigma, 1e-9)
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def two_gaussian_plus_constant(x, a0, m0, s0, a1, m1, s1, c):
    return gaussian(x, a0, m0, s0) + gaussian(x, a1, m1, s1) + c


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


def scan_integration_windows(
    files,
    baseline_window_ns,
    starts_ns,
    stops_ns,
    channel: str = "Channel 1",
    chunk_size: int = 512,
    min_width_ns: float = 1.0,
):
    windows = [(float(start), float(stop)) for start in starts_ns for stop in stops_ns if stop - start >= min_width_ns]
    charges_by_window = compute_charges_for_windows(
        files,
        baseline_window_ns,
        windows,
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


def spe_poisson_model(
    x,
    n_total,
    pedestal,
    sigma_ped,
    spe_area,
    sigma_spe,
    mu_led,
    bin_width,
    max_pe: int = 8,
):
    """Pedestal plus Poisson-weighted Gaussian photoelectron peaks."""
    y = np.zeros_like(x, dtype=float)
    mu_led = np.maximum(mu_led, 1e-9)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_led + n_pe * np.log(mu_led) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = pedestal + n_pe * spe_area
        sigma = np.sqrt(sigma_ped**2 + n_pe * sigma_spe**2)
        y += n_total * bin_width * weight / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -0.5 * ((x - mean) / sigma) ** 2
        )
    return y


def spe_poisson_components(
    x,
    n_total,
    pedestal,
    sigma_ped,
    spe_area,
    sigma_spe,
    mu_led,
    bin_width,
    max_pe: int = 8,
):
    """Return the individual n-photoelectron Gaussian components."""
    components = {}
    mu_led = np.maximum(mu_led, 1e-9)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_led + n_pe * np.log(mu_led) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = pedestal + n_pe * spe_area
        sigma = np.sqrt(sigma_ped**2 + n_pe * sigma_spe**2)
        components[n_pe] = n_total * bin_width * weight / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -0.5 * ((x - mean) / sigma) ** 2
        )

    return components


def default_fit_range(charge_mV_ns, low_quantile=0.001, high_quantile=0.98, pad_fraction=0.05):
    low, high = np.quantile(charge_mV_ns, [low_quantile, high_quantile])
    pad = pad_fraction * (high - low)
    return float(low - pad), float(high + pad)


def fit_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 8,
    min_spe_area_mV_ns: float = 1e-4,
):
    """Fit the charge spectrum with a Poisson-weighted SPE model."""
    charge_mV_ns = np.asarray(charge_mV_ns, dtype=float)
    if fit_range is None:
        fit_range = default_fit_range(charge_mV_ns)

    counts, edges = np.histogram(charge_mV_ns, bins=bins, range=fit_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    rough = score_charge_spectrum(charge_mV_ns)
    pedestal0 = rough["rough_pedestal_mV_ns"]
    sigma_ped0 = max(rough["rough_pedestal_sigma_mV_ns"], np.std(charge_mV_ns[charge_mV_ns < np.quantile(charge_mV_ns, 0.6)]) * 0.5)
    spe_area0 = rough["rough_spe_area_mV_ns"]
    if not np.isfinite(spe_area0) or spe_area0 <= 0:
        spe_area0 = max(np.quantile(charge_mV_ns, 0.95) - pedestal0, sigma_ped0 * 3)
    sigma_spe0 = max(spe_area0 * 0.4, sigma_ped0)

    p0 = [
        len(charge_mV_ns),
        pedestal0,
        sigma_ped0,
        spe_area0,
        sigma_spe0,
        0.03,
    ]
    p0[3] = max(p0[3], min_spe_area_mV_ns * 1.5)
    bounds = (
        [0.5 * len(charge_mV_ns), fit_range[0], 1e-4, min_spe_area_mV_ns, 1e-4, 0.001],
        [2.0 * len(charge_mV_ns), fit_range[1], np.inf, np.inf, np.inf, 5.0],
    )

    def model_for_fit(x, n_total, pedestal, sigma_ped, spe_area, sigma_spe, mu_led):
        return spe_poisson_model(
            x,
            n_total,
            pedestal,
            sigma_ped,
            spe_area,
            sigma_spe,
            mu_led,
            bin_width,
            max_pe=max_pe,
        )

    sigma_counts = np.sqrt(np.maximum(counts, 1))
    popt, pcov = curve_fit(
        model_for_fit,
        centers,
        counts,
        p0=p0,
        bounds=bounds,
        sigma=sigma_counts,
        absolute_sigma=True,
        maxfev=50000,
    )
    model_counts = model_for_fit(centers, *popt)
    chi2 = np.sum(((counts - model_counts) / sigma_counts) ** 2)
    ndof = len(counts) - len(popt)

    names = ["n_total", "pedestal_mV_ns", "sigma_pedestal_mV_ns", "spe_area_mV_ns", "sigma_spe_mV_ns", "mu_led"]
    parameters = dict(zip(names, popt))
    errors = dict(zip(names, np.sqrt(np.diag(pcov))))
    diagnostics = []
    if np.isclose(parameters["spe_area_mV_ns"], bounds[0][3], rtol=0, atol=max(1e-3, 0.02 * min_spe_area_mV_ns)):
        diagnostics.append("spe_area_mV_ns is close to the lower fit bound")
    if np.isclose(parameters["mu_led"], bounds[0][-1], rtol=0, atol=3e-4):
        diagnostics.append("mu_led is close to the lower fit bound")
    if np.isclose(parameters["mu_led"], bounds[1][-1], rtol=0, atol=1e-2):
        diagnostics.append("mu_led is close to the upper fit bound")

    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "bin_width": bin_width,
        "fit_range": fit_range,
        "parameters": parameters,
        "errors": errors,
        "covariance": pcov,
        "model_counts": model_counts,
        "chi2": float(chi2),
        "ndof": int(ndof),
        "diagnostics": diagnostics,
    }


def print_spe_fit_result(fit):
    for name, value in fit["parameters"].items():
        err = fit["errors"].get(name, np.nan)
        print(f"{name}: {value:.6g} +/- {err:.2g}")
    print(f"chi2/ndof: {fit['chi2']:.1f}/{fit['ndof']} = {fit['chi2'] / fit['ndof']:.3f}")
    for diagnostic in fit["diagnostics"]:
        print(f"diagnostic: {diagnostic}")


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


def plot_average_waveforms(time_ns, average_results, ax=None):
    import matplotlib.pyplot as plt

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
