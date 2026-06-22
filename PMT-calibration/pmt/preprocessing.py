"""PMT waveform preprocessing helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths
import pandas as pd


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
        "is_saturated": saturated,
        "n_saturated_samples": n_saturated_samples,
    }
    summary = {
        "n_total": int(len(keep_mask)),
        "n_kept": int(np.sum(keep_mask)),
        "n_removed": int(np.sum(saturated)),
        "efficiency": float(np.mean(keep_mask)) if len(keep_mask) else np.nan,
    }
    return voltage_mV[keep_mask], event_info, summary

# ----------------------------------------------------------------
# Baseline 
# ----------------------------------------------------------------

def subtract_baseline(time_ns, voltage_mV, baseline_window_ns=None, baseline_window=None):
    """baseline is the mean wavefrom in baseline_window_ns."""

    if baseline_window_ns is None:
        baseline_window_ns = baseline_window
    if baseline_window_ns is None:
        baseline_window_ns = (0.0, 20.0)

    mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    if not np.any(mask):
        raise ValueError(f"Baseline window {baseline_window_ns} contains no samples")
    
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

    # Baseline subtraction
    voltage_bs_mV = ( voltage_mV - baseline_mean_mV[:, None])

    event_info = {
        "baseline_mean_mV": baseline_mean_mV,
        "baseline_rms_mV": baseline_rms_mV,
        "baseline_slope_mV_per_ns": baseline_slope_mV_per_ns
    }

    return voltage_bs_mV, event_info



# ----------------------------------------------------------------
# Waveform analysis 
# ----------------------------------------------------------------


def extract_waveform_features(
    time_ns,
    voltage_mV,
    peak_threshold_mV=None,
    second_peak_min_height_frac=0.2,
    polarity = "negative"
):
    """
    Extract timing information from a single waveform.

    Parameters
    ----------
    time_ns : array-like
        Time axis, shape (n_samples,).
    voltage_mV : array-like
        Baseline-subtracted waveform, shape (n_samples,).
    peak_threshold_mV : float or None
        Minimum peak height. If None, use 10% of the main peak.
    second_peak_min_height_frac : float
        Secondary peaks must exceed this fraction of the main peak.

    Returns
    -------
    dict
    """

    t = np.asarray(time_ns)
    v = np.asarray(voltage_mV)

    # ------------------------------
    # Sanity checks
    # ------------------------------

    if t.ndim != 1:
        raise ValueError(
            f"time_ns must be 1D, got shape {t.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "pulse_timing_info expects ONE waveform.\n"
            f"Got voltage_mV.shape = {v.shape}\n"
            "Select a single event first, e.g.\n"
            "    voltage_sample_bs_mV[event_id]"
        )

    if len(t) != len(v):
        raise ValueError(
            f"time axis length ({len(t)}) "
            f"!= waveform length ({len(v)})"
        )

    if polarity == "negative":
        v = -v
    else:
        v = v
    # ---------- Main peak ----------

    peak_idx = np.argmax(v)
    peak_time_ns = t[peak_idx]
    peak_amplitude_mV = v[peak_idx]

    if peak_threshold_mV is None:
        peak_threshold_mV = 0.1 * peak_amplitude_mV

    # ---------- Rise time ----------

    level10 = 0.10 * peak_amplitude_mV
    level90 = 0.90 * peak_amplitude_mV

    before_peak = np.arange(peak_idx + 1)

    idx10 = np.where(v[before_peak] >= level10)[0]
    idx90 = np.where(v[before_peak] >= level90)[0]

    t10 = t[idx10[0]] if len(idx10) else np.nan
    t90 = t[idx90[0]] if len(idx90) else np.nan

    rise_time_ns = t90 - t10

    # ---------- Fall time ----------

    after_peak = np.arange(peak_idx, len(v))

    idx90_fall = np.where(v[after_peak] <= level90)[0]
    idx10_fall = np.where(v[after_peak] <= level10)[0]

    t90_fall = (
        t[after_peak[idx90_fall[0]]]
        if len(idx90_fall)
        else np.nan
    )

    t10_fall = (
        t[after_peak[idx10_fall[0]]]
        if len(idx10_fall)
        else np.nan
    )

    fall_time_ns = t10_fall - t90_fall

    # ---------- FWHM ----------

    halfmax = 0.5 * peak_amplitude_mV

    above_half = np.where(v >= halfmax)[0]

    if len(above_half):
        fwhm_ns = t[above_half[-1]] - t[above_half[0]]
    else:
        fwhm_ns = np.nan

    # ---------- Peak width (scipy) ----------

    try:
        widths, width_heights, left_ips, right_ips = peak_widths(
            v,
            [peak_idx],
            rel_height=0.5,
        )

        dt = np.mean(np.diff(t))

        peak_width_ns = widths[0] * dt

    except Exception:
        peak_width_ns = np.nan
    
    # ---------- Integral ----------

    area_mV_ns = np.trapezoid(v, t)

    # ---------- Peak finding ----------

    peaks, props = find_peaks(
        v,
        height=max(
            peak_threshold_mV,
            second_peak_min_height_frac * peak_amplitude_mV,
        ),
    )

    peak_heights = props["peak_heights"]

    # sort peaks by amplitude (largest first)
    order = np.argsort(peak_heights)[::-1]

    second_peak_time_ns = np.nan
    second_peak_amplitude_mV = np.nan
    peak_separation_ns = np.nan

    if len(order) > 1:
        second_idx = peaks[order[1]]
        second_peak_time_ns = t[second_idx]
        second_peak_amplitude_mV = v[second_idx]

        peak_separation_ns = (
            second_peak_time_ns - peak_time_ns
        )

    return {
        "peak_time_ns": float(peak_time_ns),
        "peak_idx": int(peak_idx),
        "peak_amplitude_mV": float(peak_amplitude_mV),
        "t10_rise_ns": float(t10), 
        "t90_rise_ns": float(t90),
        "rise_time_10_90_ns": float(rise_time_ns),
        "t10__fall_ns": float(t10_fall),
        "t90__fall_ns": float(t90_fall),
        "fall_time_90_10_ns": float(fall_time_ns),
        "fwhm_ns": float(fwhm_ns),

        "peak_width_ns": float(peak_width_ns),
        "area_mV_ns": float(area_mV_ns),
        "n_peaks": int(len(peaks)),

        "second_peak_time_ns": float(second_peak_time_ns),
        "second_peak_amplitude_mV": float(second_peak_amplitude_mV),
        "peak_separation_ns": float(peak_separation_ns),
    }



def build_waveform_feature_dataframe(
    time_ns,
    waveforms_mV,
    peak_threshold_mV=None,
    second_peak_min_height_frac=0.2,
    extra=None # extra dictionary to be added to the dataframe
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

    waveforms_mV = np.asarray(waveforms_mV)

    if waveforms_mV.ndim != 2:
        raise ValueError(
            f"Expected shape (n_events, n_samples), got {waveforms_mV.shape}"
        )

    results = []

    for event_id, waveform in enumerate(waveforms_mV):

        info = extract_waveform_features(
            time_ns=time_ns,
            voltage_mV=waveform,
            peak_threshold_mV=peak_threshold_mV,
            second_peak_min_height_frac=second_peak_min_height_frac
        )

        info["event_id"] = event_id

        results.append(info)

    df = pd.DataFrame(results)
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

    return df

# ----------------------------------------------------------------
# Selection wrappers 
# ----------------------------------------------------------------




__all__ = ["subtract_baseline",
           "remove_saturated_waveforms", 
           "build_waveform_feature_dataframe"]



