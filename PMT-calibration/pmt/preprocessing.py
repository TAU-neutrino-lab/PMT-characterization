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

def baseline_subtraction(time_ns, voltage_mV, baseline_window_ns=None):
    """baseline is the mean wavefrom in baseline_window_ns."""

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

def _fractional_crossing_time( t, v, fraction, rising=True):
    """
    compute crossing time with interpolation between the bins
    CAUTION: v must be selected according to polarity (i.e. the peak is positive)"""
    t = np.asarray(t)
    v = np.asarray(v)

    peak_idx = np.argmax(v)
    peak_amplitude = v[peak_idx]

    threshold = fraction * peak_amplitude

    if rising:
        region_t = t[:peak_idx + 1]
        region_v = v[:peak_idx + 1]

        idx = np.where(region_v >= threshold)[0]

    else:
        region_t = t[peak_idx:]
        region_v = v[peak_idx:]

        idx = np.where(region_v <= threshold)[0]

    if len(idx) == 0:
        return np.nan

    i = idx[0]

    if i == 0:
        return region_t[0]

    t1 = region_t[i - 1]
    t2 = region_t[i]

    y1 = region_v[i - 1]
    y2 = region_v[i]

    if y2 == y1:
        return t1

    return t1 + ( (threshold - y1) * (t2 - t1) / (y2 - y1) )

# ----------------------------------------------------------------
# Waveform analysis 
# ----------------------------------------------------------------

def _extract_waveform_features_fast( time_ns, waveforms_mV, polarity="negative", pre_peak_ns = 20, post_peak_ns = 80): 

    t = np.asarray(time_ns)
    v = np.asarray(waveforms_mV)
    n_events = len(v)

    if polarity == "negative":
        v = -v
    else:
        v = v

    # ---------- Peak ----------

    peak_idx = np.argmax( v, axis=1 )
    peak_amplitude = v[ np.arange(n_events), peak_idx ]
    peak_time = t[ peak_idx ]

    # ---------- Charge ----------

    area = np.trapezoid( v, t, axis=1 )

    charge_peak_window = np.empty(n_events)
    for i, peak_time_i in enumerate(peak_time):
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

def _extract_waveform_features_slow( time_ns, waveforms_mV, peak_idx, peak_amplitude_mV,
    peak_threshold_mV=None,
    second_peak_min_height_frac=0.2,
    polarity="negative",
    pre_rise_ns = 10, 
    post_rise_ns = 80, 
):

    n_events = len(waveforms_mV)
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
    for i, waveform in enumerate(waveforms_mV):

        v = waveform
        pk = peak_idx[i]
        amp = peak_amplitude_mV[i]

        if polarity == "negative":
            v = -v
        else:
            v = v

        # ---------- CFD ----------

        for frac in (10, 20, 30, 50, 90):

            tr = _fractional_crossing_time( time_ns, v, fraction=frac / 100, rising=True )
            tf = _fractional_crossing_time( time_ns, v, fraction=frac / 100, rising=False)
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

        # ---------- Peak Finding ----------

        threshold = ( 0.1 * amp if peak_threshold_mV is None else peak_threshold_mV )
        peaks, props = find_peaks( v, height=max( threshold, second_peak_min_height_frac * amp ) )
        results["n_peaks"][i] = len(peaks)

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
    second_peak_min_height_frac=0.2,
    pre_peak_ns = 20,
    post_peak_ns = 80,
    pre_rise_ns = 10, 
    post_rise_ns = 80, 
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
    
    fast = _extract_waveform_features_fast( time_ns, waveforms_mV, polarity, pre_peak_ns, post_peak_ns)
    peak_idx=fast["peak_idx"]
    peak_amplitude_mV=fast["peak_amplitude_mV"]
    slow = _extract_waveform_features_slow( time_ns=time_ns, waveforms_mV=waveforms_mV,  peak_idx=peak_idx,  peak_amplitude_mV=peak_amplitude_mV, 
                                           peak_threshold_mV=peak_threshold_mV, second_peak_min_height_frac=second_peak_min_height_frac,
                                           polarity=polarity, pre_rise_ns=pre_rise_ns, post_rise_ns=post_rise_ns )

    features = {}
    features.update(fast)
    features.update(slow)
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

    return df

# ----------------------------------------------------------------
# Selection wrappers 
# ----------------------------------------------------------------




__all__ = ["baseline_subtraction",
           "remove_saturated_waveforms", 
           "build_waveform_feature_dataframe",

           "_fractional_crossing_time"
           ]



