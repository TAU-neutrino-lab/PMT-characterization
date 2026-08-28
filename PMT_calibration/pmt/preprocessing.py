"""PMT waveform preprocessing helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths
import pandas as pd

"""
    Builds a pandas dataframe
"""


# ----------------------------------------------------------------
# Saturated waveforms
# CAUTION: must run BEFORE baseline subtraction
# ----------------------------------------------------------------

def _metadata_voltage_limits_mV(metadata):
    attrs = metadata.get("channel_attrs", metadata)
    if "YDispOrigin" not in attrs or "YDispRange" not in attrs:
        return None, None
    center_mV = float(attrs["YDispOrigin"]) * 1e3
    half_range_mV = float(attrs["YDispRange"]) * 1e3 / 2.0
    return center_mV - half_range_mV, center_mV + half_range_mV

def saturated_waveform_mask( voltage_mV, low_limit_mV=None, high_limit_mV=None, 
    margin_mV: float = 0.0, # (low_limit_mV + margin_mV) will count as saturated
    max_saturated_samples: int = 0, # if there are more saturated samples than max_saturated_samples then the waveform is saturated
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

def remove_saturated_waveforms( voltage_mV, metadata, low_limit_mV=None, high_limit_mV=None, margin_mV: float = 0.0, max_saturated_samples: int = 0):

    if low_limit_mV is None and high_limit_mV is None:
        low_limit_mV, high_limit_mV = _metadata_voltage_limits_mV(metadata) 

    """Remove saturated raw waveforms and return ``selected, keep_mask, info``."""
    saturated, n_saturated_samples = saturated_waveform_mask(
        voltage_mV,
        low_limit_mV=low_limit_mV,
        high_limit_mV=high_limit_mV,
        margin_mV=margin_mV,
        max_saturated_samples=max_saturated_samples,
    )
    keep_mask = ~saturated
    event_info = {
        "is_saturated":        saturated,
        "n_saturated_samples": n_saturated_samples,
    }
    summary = {
        "n_total":    int(len(keep_mask)),
        "n_kept":     int(np.sum(keep_mask)),
        "n_removed":  int(np.sum(saturated)),
        "efficiency": float(np.mean(keep_mask)) if len(keep_mask) else np.nan,
    }
    return voltage_mV[keep_mask], keep_mask, event_info, summary

# ----------------------------------------------------------------
# Baseline 
# ----------------------------------------------------------------

def baseline_cleanliness_mask(
    time_ns,
    voltage_mV,
    baseline_window_ns=(0.0, 20.0),
    excursion_snr=8.0,
    polarity="negative",
):
    """Return a mask selecting raw waveforms with a pulse-free baseline window.

    Median and MAD estimates are deliberately computed before baseline
    subtraction, so a pulse in the window cannot inflate the mean/RMS used to
    detect itself.
    """
    time_ns = np.asarray(time_ns)
    voltage_mV = np.asarray(voltage_mV)
    window = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if not np.any(window):
        raise ValueError(f"Baseline window {baseline_window_ns} contains no samples")

    region = voltage_mV[:, window]
    center = np.median(region, axis=1)
    mad = np.median(np.abs(region - center[:, None]), axis=1)
    robust_rms = 1.4826 * mad
    fallback_rms = np.std(region, axis=1)
    robust_rms = np.where(robust_rms > 0, robust_rms, fallback_rms)

    if polarity == "negative":
        excursion = center - np.min(region, axis=1)
    elif polarity == "positive":
        excursion = np.max(region, axis=1) - center
    else:
        raise ValueError("polarity must be 'negative' or 'positive'")

    excursion_snr_values = np.divide(
        excursion,
        robust_rms,
        out=np.full(excursion.shape, np.inf),
        where=robust_rms > 0,
    )
    clean = excursion_snr_values < float(excursion_snr)
    info = {
        "baseline_window_is_clean": clean,
        "baseline_robust_rms_mV": robust_rms,
        "baseline_max_excursion_mV": excursion,
        "baseline_max_excursion_snr": excursion_snr_values,
    }
    return clean, info


def baseline_subtraction(
    time_ns,
    voltage_mV,
    baseline_window_ns=None,
    baseline_reference_time_ns=None,
    baseline_reference_mV=None,
):
    """Subtract a baseline template and the remaining per-event DC offset.

    Without a reference, this retains the historical behavior of subtracting
    each waveform's mean in ``baseline_window_ns``.  When a reference is
    supplied, its pointwise waveform is subtracted first, followed by the mean
    residual in the baseline window.  The latter protects charge integrals
    against run-to-run DC drift relative to the saved reference acquisition.
    """

    if baseline_window_ns is None:
        baseline_window_ns = (0.0, 20.0)

    mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if not np.any(mask):
        raise ValueError(f"Baseline window {baseline_window_ns} contains no samples")
    
    use_reference = baseline_reference_mV is not None
    if use_reference != (baseline_reference_time_ns is not None):
        raise ValueError(
            "baseline_reference_time_ns and baseline_reference_mV must be supplied together"
        )
    if use_reference:
        reference_time_ns = np.asarray(baseline_reference_time_ns, dtype=float)
        reference_mV = np.asarray(baseline_reference_mV, dtype=float)
        if reference_time_ns.shape != np.asarray(time_ns).shape:
            raise ValueError("Baseline-reference time axis has the wrong shape")
        if reference_mV.shape != np.asarray(time_ns).shape:
            raise ValueError("Baseline-reference waveform has the wrong shape")
        if not np.allclose(reference_time_ns, time_ns, rtol=1e-7, atol=1e-9):
            raise ValueError("Baseline-reference time axis does not match the acquisition")
    else:
        reference_mV = np.zeros_like(time_ns, dtype=float)

    #--- statistics
    baseline_region = voltage_mV[:, mask]
    # Mean baseline
    baseline_mean_mV = np.mean( baseline_region, axis=1)
    # RMS noise
    baseline_rms_mV = np.std(baseline_region, axis=1)
    # Baseline slope
    baseline_t = time_ns[mask]
    baseline_slope_mV_per_ns = np.full( voltage_mV.shape[0], np.nan )
    if len(baseline_t) >= 2:
        x = baseline_t
        xmean = np.mean(x)
        denom = np.sum((x - xmean) ** 2)
        ymean = np.mean( baseline_region, axis=1)
        numer = np.sum( (x - xmean)[None, :] * (baseline_region - ymean[:, None]),  axis=1 )
        baseline_slope_mV_per_ns = numer / denom

    # Subtract coherent reference structure, then remove the remaining scalar
    # offset for each event. With a zero reference this is the legacy mean-only
    # subtraction exactly.
    reference_subtracted_mV = voltage_mV - reference_mV[None, :]
    residual_baseline_mean_mV = np.mean(reference_subtracted_mV[:, mask], axis=1)
    voltage_bs_mV = reference_subtracted_mV - residual_baseline_mean_mV[:, None]

    event_info = {
        "baseline_mean_mV": baseline_mean_mV,
        "baseline_rms_mV": baseline_rms_mV,
        "baseline_slope_mV_per_ns": baseline_slope_mV_per_ns,
        "baseline_reference_residual_mean_mV": residual_baseline_mean_mV,
    }

    return voltage_bs_mV, event_info

# ----------------------------------------------------------------
# Charge computation 
# ----------------------------------------------------------------

def _time_window_indices( time_ns, center_time_ns, pre_ns, post_ns ):
    """
    Return the indices corresponding to a time window.

    Parameters
    ----------
    time_ns : (n_samples,)
        Time axis.

    center_time_ns : float
        Center of the window.

    pre_ns, post_ns : float
        Time before and after the center.

    Returns
    -------
    lo, hi : int
        Slice indices such that

            time_ns[lo:hi]

        corresponds to

            center_time_ns-pre_ns <= t <= center_time_ns+post_ns
    """

    lo = np.searchsorted( time_ns, center_time_ns - pre_ns, side="left")
    hi = np.searchsorted( time_ns, center_time_ns + post_ns, side="right" )

    return lo, hi

def _charge_around_time( time_ns, waveform_mV, center_time_ns, pre_ns, post_ns ):

    lo, hi = _time_window_indices( time_ns, center_time_ns, pre_ns, post_ns )

    if hi <= lo:
        return np.nan

    return np.trapezoid( waveform_mV[lo:hi], time_ns[lo:hi] )

def integrate_led_window_charge(
    time_ns,
    waveforms_mV,
    led_time_ns,
    pre_led_ns=20.0,
    post_led_ns=80.0,
    polarity="negative",
    require_full_window=False,
):
    """Integrate every waveform in one LED-synchronous time window.

    Unlike ``charge_peak_window_mV_ns``, the window is not moved to the
    detected maximum of each event.  This keeps pedestal and low-SNR events
    unbiased by noise-peak finding and gives every event the same exposure to
    baseline noise. If the requested bounds extend beyond the recorded trace,
    the integral is clipped unless ``require_full_window`` is true.
    """
    time_ns = np.asarray(time_ns)
    waveforms_mV = np.asarray(waveforms_mV)

    if time_ns.ndim != 1:
        raise ValueError("time_ns must be one-dimensional")
    if waveforms_mV.ndim != 2 or waveforms_mV.shape[1] != len(time_ns):
        raise ValueError("waveforms_mV must have shape (n_events, n_samples)")
    if pre_led_ns < 0 or post_led_ns < 0:
        raise ValueError("pre_led_ns and post_led_ns must be non-negative")
    if pre_led_ns + post_led_ns <= 0:
        raise ValueError("the LED integration window must have positive duration")

    requested_start_ns = led_time_ns - pre_led_ns
    requested_stop_ns = led_time_ns + post_led_ns
    if require_full_window and (
        requested_start_ns < time_ns[0] or requested_stop_ns > time_ns[-1]
    ):
        raise ValueError(
            f"LED charge window [{requested_start_ns:g}, {requested_stop_ns:g}] ns "
            f"is not contained in the waveform [{time_ns[0]:g}, {time_ns[-1]:g}] ns"
        )

    # searchsorted naturally clips the requested bounds to the recorded array.
    # The caller can inspect ``charge_led_window_coverage`` from the feature
    # dataframe to distinguish a complete from a truncated integral.
    lo, hi = _time_window_indices(
        time_ns, led_time_ns, pre_led_ns, post_led_ns
    )
    if hi <= lo:
        return np.full(waveforms_mV.shape[0], np.nan)
    pulse_positive_mV = -waveforms_mV if polarity == "negative" else waveforms_mV
    return np.trapezoid(
        pulse_positive_mV[:, lo:hi], time_ns[lo:hi], axis=1
    )

def _fractional_crossing_time(t, v, fraction, rising=True, peak_idx=None):
    """
    compute crossing time with interpolation between the bins
    CAUTION: v must be selected according to polarity (i.e. the peak is positive)"""
    t = np.asarray(t)
    v = np.asarray(v)

    if peak_idx is None:
        peak_idx = np.argmax(v)
    peak_amplitude = v[peak_idx]

    threshold = fraction * peak_amplitude

    if rising:
        region_t = t[:peak_idx + 1]
        region_v = v[:peak_idx + 1]
        # Use the last upward crossing before the peak. Searching for the first
        # sample above a low CFD threshold can lock onto an unrelated baseline
        # fluctuation long before the actual pulse.
        crossings = np.where(
            (region_v[:-1] < threshold) & (region_v[1:] >= threshold)
        )[0]
        if len(crossings):
            i = int(crossings[-1] + 1)
        elif region_v[0] >= threshold:
            return float(region_t[0])
        else:
            return np.nan
    else:
        region_t = t[peak_idx:]
        region_v = v[peak_idx:]
        crossings = np.where(
            (region_v[:-1] > threshold) & (region_v[1:] <= threshold)
        )[0]
        if len(crossings):
            i = int(crossings[0] + 1)
        elif region_v[0] <= threshold:
            return float(region_t[0])
        else:
            return np.nan

    t1 = region_t[i - 1]
    t2 = region_t[i]

    y1 = region_v[i - 1]
    y2 = region_v[i]

    if y2 == y1:
        return t1

    return t1 + ( (threshold - y1) * (t2 - t1) / (y2 - y1) )


def learn_fixed_pulse_charge_window(
    df,
    *,
    reference_snr=15.0,
    width_quantile=0.95,
    min_reference_pulses=100,
    generic_shape_ranges=None,
    max_additional_peak_fraction=0.75,
):
    """Learn one fixed integration window from clean single-pulse events.

    The window width is the requested quantile of each reference pulse's
    CFD-10 falling-minus-rising duration. Its center is the median CFD-10
    midpoint. The learned bounds are trigger-relative and can therefore be
    applied unchanged to pulse and pedestal events.
    """

    required_columns = {
        "snr",
        "n_peaks",
        "n_signal_like_peaks",
        "largest_additional_signal_like_peak_fraction",
        "cfd10_rise_ns",
        "cfd10_fall_ns",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Fixed pulse-window learning requires missing dataframe columns: "
            f"{sorted(missing_columns)}"
        )

    reference_snr = float(reference_snr)
    width_quantile = float(width_quantile)
    min_reference_pulses = int(min_reference_pulses)
    if not np.isfinite(reference_snr):
        raise ValueError("reference_snr must be finite")
    if not 0.0 < width_quantile <= 1.0:
        raise ValueError("width_quantile must be in (0, 1]")
    if min_reference_pulses < 1:
        raise ValueError("min_reference_pulses must be at least 1")
    max_additional_peak_fraction = float(max_additional_peak_fraction)
    if not 0.0 <= max_additional_peak_fraction <= 1.0:
        raise ValueError("max_additional_peak_fraction must be between 0 and 1")

    if generic_shape_ranges is None:
        generic_shape_ranges = {}
    else:
        generic_shape_ranges = dict(generic_shape_ranges)
    supported_shape_columns = {
        "peak_width_ns", "rise_time_10_90_ns", "fall_time_90_10_ns"
    }
    unknown_shape_columns = set(generic_shape_ranges).difference(
        supported_shape_columns
    )
    if unknown_shape_columns:
        raise ValueError(
            "generic_shape_ranges contains unsupported columns: "
            f"{sorted(unknown_shape_columns)}"
        )
    generic_shape_mask = np.ones(len(df), dtype=bool)
    for column, bounds in generic_shape_ranges.items():
        if column not in df.columns:
            raise ValueError(
                f"Fixed pulse-window learning requires missing dataframe column: {column}"
            )
        if len(bounds) != 2:
            raise ValueError(f"Generic bounds for {column} must have two values")
        low, high = map(float, bounds)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(
                f"Generic bounds for {column} must be finite with low < high"
            )
        generic_shape_ranges[column] = (low, high)
        generic_shape_mask &= df[column].between(low, high).fillna(False).to_numpy()

    rise_ns = df["cfd10_rise_ns"].to_numpy(dtype=float)
    fall_ns = df["cfd10_fall_ns"].to_numpy(dtype=float)
    durations_ns = fall_ns - rise_ns
    has_dominant_peak = (
        df["n_signal_like_peaks"].eq(1)
        | (
            df["n_signal_like_peaks"].gt(1)
            & df["largest_additional_signal_like_peak_fraction"].le(
                max_additional_peak_fraction
            ).fillna(False)
        )
    ).to_numpy()
    reference_mask = (
        df["snr"].ge(reference_snr).fillna(False).to_numpy()
        & df["n_peaks"].eq(1).to_numpy()
        & has_dominant_peak
        & generic_shape_mask
        & np.isfinite(rise_ns)
        & np.isfinite(fall_ns)
        & (durations_ns > 0.0)
    )
    reference_count = int(np.sum(reference_mask))
    if reference_count < min_reference_pulses:
        raise ValueError(
            "Cannot learn the fixed pulse charge window: found "
            f"{reference_count} valid dominant single pulses at "
            f"SNR >= {reference_snr:g}, "
            f"but at least {min_reference_pulses} are required."
        )

    reference_durations_ns = durations_ns[reference_mask]
    reference_midpoints_ns = 0.5 * (
        rise_ns[reference_mask] + fall_ns[reference_mask]
    )
    width_ns = float(np.quantile(reference_durations_ns, width_quantile))
    center_ns = float(np.median(reference_midpoints_ns))
    start_ns = center_ns - 0.5 * width_ns
    stop_ns = center_ns + 0.5 * width_ns
    contained = (
        (rise_ns[reference_mask] >= start_ns)
        & (fall_ns[reference_mask] <= stop_ns)
    )

    return {
        "reference_snr": reference_snr,
        "width_quantile": width_quantile,
        "min_reference_pulses": min_reference_pulses,
        "reference_pulses": reference_count,
        "generic_shape_ranges": generic_shape_ranges,
        "max_additional_peak_fraction": max_additional_peak_fraction,
        "center_ns": center_ns,
        "width_ns": width_ns,
        "window_ns": (start_ns, stop_ns),
        "reference_containment_fraction": float(np.mean(contained)),
    }

# ----------------------------------------------------------------
# Waveform analysis 
# ----------------------------------------------------------------

def _find_waveform_peaks(
    waveforms_mV,
    baseline_rms_mV,
    *,
    polarity="negative",
    peak_snr_threshold=5.0,
    peak_threshold_mV=None,
    peak_prominence_snr=None,
    peak_distance_samples=None,
    peak_width_samples=None,
):
    """Find qualifying peaks in every waveform using noise-relative cuts."""
    pulse_positive_mV = -np.asarray(waveforms_mV) if polarity == "negative" else np.asarray(waveforms_mV)
    baseline_rms_mV = np.asarray(baseline_rms_mV, dtype=float)
    if baseline_rms_mV.shape != (len(pulse_positive_mV),):
        raise ValueError("baseline_rms_mV must contain one value per waveform")

    peaks_by_event = []
    properties_by_event = []
    for waveform, baseline_rms in zip(pulse_positive_mV, baseline_rms_mV):
        minimum_height = 0.0 if peak_threshold_mV is None else float(peak_threshold_mV)
        if peak_snr_threshold is not None:
            minimum_height = max(minimum_height, float(peak_snr_threshold) * baseline_rms)
        prominence = (
            None if peak_prominence_snr is None
            else float(peak_prominence_snr) * baseline_rms
        )
        peaks, properties = find_peaks(
            waveform,
            height=minimum_height,
            prominence=prominence,
            distance=peak_distance_samples,
            width=peak_width_samples,
        )
        peaks_by_event.append(peaks)
        properties_by_event.append(properties)
    return pulse_positive_mV, peaks_by_event, properties_by_event


def _summarize_signal_like_peaks(
    pulse_positive_mV,
    baseline_rms_mV,
    primary_peak_idx,
    *,
    peak_snr_threshold=5.0,
    peak_threshold_mV=None,
    peak_prominence_snr=None,
    peak_distance_samples=None,
):
    """Summarize height/prominence-qualified candidates before width cuts."""
    counts = np.zeros(len(pulse_positive_mV), dtype=int)
    largest_additional_amplitude = np.full(len(pulse_positive_mV), np.nan)
    largest_additional_idx = np.full(len(pulse_positive_mV), -1, dtype=int)
    largest_additional_fraction = np.zeros(len(pulse_positive_mV), dtype=float)
    for index, (waveform, baseline_rms) in enumerate(
        zip(pulse_positive_mV, np.asarray(baseline_rms_mV, dtype=float))
    ):
        minimum_height = 0.0 if peak_threshold_mV is None else float(peak_threshold_mV)
        if peak_snr_threshold is not None:
            minimum_height = max(
                minimum_height, float(peak_snr_threshold) * baseline_rms
            )
        prominence = (
            None if peak_prominence_snr is None
            else float(peak_prominence_snr) * baseline_rms
        )
        peaks, _ = find_peaks(
            waveform,
            height=minimum_height,
            prominence=prominence,
            distance=peak_distance_samples,
        )
        counts[index] = len(peaks)
        primary = int(primary_peak_idx[index])
        if primary < 0 or not len(peaks):
            largest_additional_fraction[index] = np.nan
            continue
        additional = peaks[peaks != primary]
        if not len(additional):
            continue
        additional_idx = int(additional[np.argmax(waveform[additional])])
        additional_amplitude = float(waveform[additional_idx])
        primary_amplitude = float(waveform[primary])
        largest_additional_idx[index] = additional_idx
        largest_additional_amplitude[index] = additional_amplitude
        largest_additional_fraction[index] = (
            additional_amplitude / primary_amplitude
            if primary_amplitude > 0.0 else np.inf
        )
    return {
        "n_signal_like_peaks": counts,
        "largest_additional_signal_like_peak_idx": largest_additional_idx,
        "largest_additional_signal_like_peak_amplitude_mV": (
            largest_additional_amplitude
        ),
        "largest_additional_signal_like_peak_fraction": (
            largest_additional_fraction
        ),
    }


def _extract_waveform_features_fast(
    time_ns,
    pulse_positive_mV,
    peaks_by_event,
    properties_by_event,
    pre_peak_ns=20,
    post_peak_ns=80,
):

    t = np.asarray(time_ns)
    v = np.asarray(pulse_positive_mV)
    n_events = len(v)

    # ---------- Peak ----------
    peak_idx = np.full(n_events, -1, dtype=int)
    peak_amplitude = np.full(n_events, np.nan)
    peak_time = np.full(n_events, np.nan)
    for i, (peaks, properties) in enumerate(zip(peaks_by_event, properties_by_event)):
        if len(peaks) == 0:
            continue
        primary_order = np.argmax(properties["peak_heights"])
        peak_idx[i] = peaks[primary_order]
        peak_amplitude[i] = properties["peak_heights"][primary_order]
        peak_time[i] = t[peak_idx[i]]

    # ---------- Charge ----------

    area = np.trapezoid( v, t, axis=1 )

    charge_peak_window = np.full(n_events, np.nan)
    for i, peak_time_i in enumerate(peak_time):
        if not np.isfinite(peak_time_i):
            continue
        lo, hi = _time_window_indices(time_ns, peak_time_i, pre_peak_ns, post_peak_ns)

        if hi <= lo:
            charge_peak_window[i] = np.nan
        else:
            charge_peak_window[i] = np.trapezoid( v[i, lo:hi], time_ns[lo:hi] )

    return {
        "peak_idx": peak_idx,
        "peak_time_ns": peak_time,
        "peak_amplitude_mV": peak_amplitude,
        "area_mV_ns": area,
        "charge_peak_window_mV_ns": charge_peak_window
    }

def _extract_waveform_features_slow(
    time_ns,
    pulse_positive_mV,
    peak_idx,
    peaks_by_event,
    properties_by_event,
    pre_rise_ns = 10, 
    post_rise_ns = 80, 
):

    n_events = len(pulse_positive_mV)
    results = {
        "peak_width_ns": np.full(n_events, np.nan),
        "rise_time_10_90_ns": np.full(n_events, np.nan),
        "fall_time_90_10_ns": np.full(n_events, np.nan),

        "cfd10_rise_ns": np.full(n_events, np.nan),
        "cfd20_rise_ns": np.full(n_events, np.nan),
        "cfd30_rise_ns": np.full(n_events, np.nan),
        "cfd50_rise_ns": np.full(n_events, np.nan),
        "cfd90_rise_ns": np.full(n_events, np.nan),

        "cfd10_fall_ns": np.full(n_events, np.nan),
        "cfd20_fall_ns": np.full(n_events, np.nan),
        "cfd30_fall_ns": np.full(n_events, np.nan),
        "cfd50_fall_ns": np.full(n_events, np.nan),
        "cfd90_fall_ns": np.full(n_events, np.nan),

        "charge_cfd10_window_mV_ns": np.full(n_events, np.nan),
        "charge_cfd20_window_mV_ns": np.full(n_events, np.nan),
        "charge_cfd30_window_mV_ns": np.full(n_events, np.nan),
        "charge_cfd50_window_mV_ns": np.full(n_events, np.nan),
        "charge_cfd90_window_mV_ns": np.full(n_events, np.nan),

        "n_peaks": np.zeros(n_events, dtype=int),

        "second_peak_time_ns": np.full(n_events, np.nan),
        "second_peak_amplitude_mV": np.full(n_events, np.nan),
        "peak_separation_ns": np.full(n_events, np.nan),
    }
    dt = np.mean(np.diff(time_ns))
    for i, v in enumerate(pulse_positive_mV):
        pk = peak_idx[i]
        peaks = peaks_by_event[i]
        props = properties_by_event[i]
        results["n_peaks"][i] = len(peaks)
        if len(peaks) == 0:
            continue

        # ---------- CFD ----------

        for frac in (10, 20, 30, 50, 90):

            tr = _fractional_crossing_time(
                time_ns, v, fraction=frac / 100, rising=True, peak_idx=pk
            )
            tf = _fractional_crossing_time(
                time_ns, v, fraction=frac / 100, rising=False, peak_idx=pk
            )
            results[f"cfd{frac}_rise_ns"][i] = tr
            results[f"cfd{frac}_fall_ns"][i] = tf

            results[f"charge_cfd{frac}_window_mV_ns"][i] = _charge_around_time( time_ns, v, tr, pre_rise_ns, post_rise_ns )

        results["rise_time_10_90_ns"][i] = ( results["cfd90_rise_ns"][i] - results["cfd10_rise_ns"][i] )
        results["fall_time_90_10_ns"][i] = ( results["cfd10_fall_ns"][i] - results["cfd90_fall_ns"][i] )
        
        # ---------- Peak Width ----------

        try:
            widths, _, _, _ = peak_widths( v, [pk], rel_height=0.5 )
            results["peak_width_ns"][i] = ( widths[0] * dt )

        except Exception:
            pass

        if len(peaks) > 1:
            order = np.argsort(props["peak_heights"])[::-1]
            second = peaks[order[1]]
            results["second_peak_time_ns"][i] = (time_ns[second])
            results["second_peak_amplitude_mV"][i] = (v[second])
            results["peak_separation_ns"][i] = ( time_ns[second] - time_ns[pk])

    return results

def build_waveform_feature_dataframe(
    time_ns,
    waveforms_mV,
    peak_threshold_mV=None,
    peak_snr_threshold=5.0,
    peak_prominence_snr=None,
    peak_distance_samples=None,
    peak_width_samples=None,
    pre_peak_ns = 20,
    post_peak_ns = 80,
    pre_rise_ns = 10, 
    post_rise_ns = 80, 
    led_time_ns=None,
    pre_led_ns=20.0,
    post_led_ns=80.0,
    extra=None, # extra dictionary to be added to the dataframe
    polarity="negative"
):
    """
    Analyze an array of waveforms.

    Parameters
    ----------
    time_ns : (n_samples,)
    waveforms_mV : (n_events, n_samples)

    Returns
    -------
    pandas.DataFrame
    """
    
    if extra is None or "baseline_rms_mV" not in extra:
        raise ValueError(
            "baseline_rms_mV is required for noise-relative peak detection"
        )
    pulse_positive_mV, peaks_by_event, properties_by_event = _find_waveform_peaks(
        waveforms_mV,
        extra["baseline_rms_mV"],
        polarity=polarity,
        peak_snr_threshold=peak_snr_threshold,
        peak_threshold_mV=peak_threshold_mV,
        peak_prominence_snr=peak_prominence_snr,
        peak_distance_samples=peak_distance_samples,
        peak_width_samples=peak_width_samples,
    )
    fast = _extract_waveform_features_fast(
        time_ns, pulse_positive_mV, peaks_by_event, properties_by_event,
        pre_peak_ns, post_peak_ns,
    )
    peak_idx=fast["peak_idx"]
    signal_like_peak_summary = _summarize_signal_like_peaks(
        pulse_positive_mV,
        extra["baseline_rms_mV"],
        peak_idx,
        peak_snr_threshold=peak_snr_threshold,
        peak_threshold_mV=peak_threshold_mV,
        peak_prominence_snr=peak_prominence_snr,
        peak_distance_samples=peak_distance_samples,
    )
    slow = _extract_waveform_features_slow(
        time_ns=time_ns,
        pulse_positive_mV=pulse_positive_mV,
        peak_idx=peak_idx,
        peaks_by_event=peaks_by_event,
        properties_by_event=properties_by_event,
        pre_rise_ns=pre_rise_ns,
        post_rise_ns=post_rise_ns,
    )

    features = {}
    features.update(fast)
    features.update(slow)
    features.update(signal_like_peak_summary)
    signal_like_peak_counts = signal_like_peak_summary["n_signal_like_peaks"]
    additional_idx = signal_like_peak_summary[
        "largest_additional_signal_like_peak_idx"
    ]
    additional_time_ns = np.full(len(additional_idx), np.nan)
    has_additional = additional_idx >= 0
    additional_time_ns[has_additional] = np.asarray(time_ns)[
        additional_idx[has_additional]
    ]
    features["largest_additional_signal_like_peak_time_ns"] = additional_time_ns
    features["signal_like_bad_shape"] = (
        (signal_like_peak_counts > 0) & (slow["n_peaks"] == 0)
    )
    max_excursion_idx = np.argmax(pulse_positive_mV, axis=1)
    features["max_excursion_time_ns"] = np.asarray(time_ns)[max_excursion_idx]
    features["max_excursion_amplitude_mV"] = pulse_positive_mV[
        np.arange(len(pulse_positive_mV)), max_excursion_idx
    ]
    if led_time_ns is not None:
        features["charge_led_window_mV_ns"] = integrate_led_window_charge(
            time_ns,
            waveforms_mV,
            led_time_ns=led_time_ns,
            pre_led_ns=pre_led_ns,
            post_led_ns=post_led_ns,
            polarity=polarity,
        )
        requested_duration_ns = pre_led_ns + post_led_ns
        recorded_start_ns = max(time_ns[0], led_time_ns - pre_led_ns)
        recorded_stop_ns = min(time_ns[-1], led_time_ns + post_led_ns)
        recorded_duration_ns = max(0.0, recorded_stop_ns - recorded_start_ns)
        features["charge_led_window_coverage"] = np.full(
            len(waveforms_mV),
            recorded_duration_ns / requested_duration_ns,
        )
    df = pd.DataFrame(features)

    # ---------- extra features ----------
    if extra is not None:
        for key, values in extra.items():
            if len(values) != len(df):
                raise ValueError(
                    f"extra['{key}'] has length {len(values)} "
                    f"but dataframe has {len(df)} rows"
                )

            df[key] = values
        if ( "baseline_rms_mV" in df.columns and "peak_amplitude_mV" in df.columns ):
            df["snr"] = ( df["peak_amplitude_mV"] / df["baseline_rms_mV"])
            df["max_excursion_snr"] = (
                df["max_excursion_amplitude_mV"] / df["baseline_rms_mV"]
            )

    return df

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def apply_cut_v(df, waveforms, variable, range=(0, 11)):
    """
     return the waveform array after applying a cut on the corresponding dataframe
        
    """
    if range[0] is None:
        mask = df[variable] <= range[1]
    elif range[1] is None:
        mask = df[variable] > range[0]
    else:
        mask = (df[variable] > range[0]) & (df[variable] <= range[1])
    return waveforms[mask.values], mask



__all__ = ["baseline_subtraction",
           "baseline_cleanliness_mask",
           "remove_saturated_waveforms", 
           "build_waveform_feature_dataframe",
           "integrate_led_window_charge",
           "learn_fixed_pulse_charge_window",
           "apply_cut_v",

           "_fractional_crossing_time"
           ]
