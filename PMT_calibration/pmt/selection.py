import numpy as np
import pandas as pd


VALID_CALIBRATION_SELECTION_MODES = {
    "standard",
    "timing_only",
    "loose_peak_multiplicity",
    "dark_counts",
}


# ----------------------------------------------------------------
# Class
# ----------------------------------------------------------------

class CutFlow:
    """
    Helper class to build a cumulative event selection.

    Example
    -------
    cutflow = CutFlow(len(df))

    cutflow.apply(
        "Baseline pulse",
        reject_baseline_pulses(...),
    )

    cutflow.apply(
        "SNR > 5",
        df["snr"] > 5,
    )

    df_sel = df[cutflow.mask].copy()
    waveforms_sel = waveforms[cutflow.mask]
    """

    def __init__(self, n_events):

        self._initial = int(n_events)
        self._mask = np.ones( n_events, dtype=bool )

        self._rows = [
            {
                "cut": "Initial",
                "n_kept": self._initial,
                "relative_efficiency": 1.0,
                "cumulative_efficiency": 1.0,
            }
        ]

    @property
    def mask(self):
        return self._mask

    def apply(self, name, mask):

        mask = np.asarray(mask)

        if mask.shape != self._mask.shape: raise ValueError( "Mask has incorrect length." )

        before = self._mask.sum()
        self._mask &= mask
        after = self._mask.sum()
        self._rows.append(
            {
                "cut": name,
                "n_kept": int(after),
                "relative_efficiency": ( after / before if before > 0 else np.nan ),
                "cumulative_efficiency": ( after / self._initial ),
            }
        )
        return self._mask

    def dataframe(self):

        return pd.DataFrame(self._rows)

    def print(self):

        df = self.dataframe()

        print( f"{'Cut':<25}" f"{'Kept':>10}" f"{'Relative':>12}" f"{'Cumulative':>14}" )
        print("-" * 62)
        for _, row in df.iterrows():
            print(
                f"{row['cut']:<25}"
                f"{row['n_kept']:>10d}"
                f"{100*row['relative_efficiency']:>11.1f}%"
                f"{100*row['cumulative_efficiency']:>13.1f}%"
            )


# ----------------------------------------------------------------
# Individual Cuts
# ----------------------------------------------------------------

def reject_baseline_pulses( df, baseline_window_ns, min_peak_height_mV=0.5 ):

    reject = ( df["peak_time_ns"].between( *baseline_window_ns ) & (df["peak_amplitude_mV"] >= min_peak_height_mV) )

    return ~reject


def add_learned_signal_classification(
    df,
    *,
    reference_snr=15.0,
    shape_quantiles=(0.01, 0.99),
    timing_quantiles=(0.01, 0.99),
    min_reference_pulses=100,
    generic_shape_ranges=None,
    max_additional_peak_fraction=0.75,
    copy=True,
):
    """Add pulse-shape and trigger-alignment classes learned from clean pulses.

    Bounds are learned from finite, exactly-one-peak events at or above
    ``reference_snr`` that also have exactly one height/prominence-qualified
    candidate before the pulse-width cut, unless every additional candidate
    is small compared with the primary pulse. Optional ``generic_shape_ranges``
    impose acquisition-independent bounds before learning. QDC is deliberately
    not used, keeping charge independent of the classification being tested.
    Set ``copy=False`` when the caller owns ``df`` and wants to append the
    classification columns in place to avoid a full-table copy.
    """
    shape_columns = (
        "peak_width_ns",
        "rise_time_10_90_ns",
        "fall_time_90_10_ns",
    )
    required = {
        "n_peaks", "n_signal_like_peaks", "snr", "peak_time_ns",
        "largest_additional_signal_like_peak_fraction",
        *shape_columns,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Learned signal classification requires columns: {sorted(missing)}"
        )
    shape_quantiles = tuple(map(float, shape_quantiles))
    timing_quantiles = tuple(map(float, timing_quantiles))
    for name, quantiles in (
        ("shape_quantiles", shape_quantiles),
        ("timing_quantiles", timing_quantiles),
    ):
        if len(quantiles) != 2 or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0:
            raise ValueError(f"{name} must satisfy 0 <= low < high <= 1")
    max_additional_peak_fraction = float(max_additional_peak_fraction)
    if not 0.0 <= max_additional_peak_fraction <= 1.0:
        raise ValueError("max_additional_peak_fraction must be between 0 and 1")

    if generic_shape_ranges is None:
        generic_shape_ranges = {}
    else:
        generic_shape_ranges = dict(generic_shape_ranges)
    unknown_generic_columns = set(generic_shape_ranges).difference(shape_columns)
    if unknown_generic_columns:
        raise ValueError(
            "generic_shape_ranges contains unsupported columns: "
            f"{sorted(unknown_generic_columns)}"
        )
    for column, bounds in generic_shape_ranges.items():
        if len(bounds) != 2:
            raise ValueError(f"Generic bounds for {column} must have two values")
        low, high = map(float, bounds)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(
                f"Generic bounds for {column} must be finite with low < high"
            )
        generic_shape_ranges[column] = (low, high)

    finite_columns = (*shape_columns, "peak_time_ns")
    is_single_peak = df["n_peaks"].eq(1)
    has_isolated_peak = is_single_peak & df["n_signal_like_peaks"].eq(1)
    has_dominant_peak = is_single_peak & (
        df["n_signal_like_peaks"].eq(1)
        | (
            df["n_signal_like_peaks"].gt(1)
            & df["largest_additional_signal_like_peak_fraction"].le(
                max_additional_peak_fraction
            ).fillna(False)
        )
    )
    has_generic_pulse_shape = is_single_peak.copy()
    for column, bounds in generic_shape_ranges.items():
        has_generic_pulse_shape &= df[column].between(*bounds).fillna(False)
    reference_mask = (
        has_dominant_peak
        & has_generic_pulse_shape
        & df["snr"].ge(float(reference_snr))
        & np.isfinite(df.loc[:, finite_columns]).all(axis=1)
    )
    reference = df.loc[reference_mask]
    if len(reference) < int(min_reference_pulses):
        raise ValueError(
            "Cannot learn signal classification: found "
            f"{len(reference)} finite dominant single peaks at "
            f"SNR >= {reference_snr:g}, "
            f"but {int(min_reference_pulses)} are required"
        )

    shape_ranges = {
        column: tuple(
            float(value)
            for value in reference[column].quantile(shape_quantiles)
        )
        for column in shape_columns
    }
    timing_range_ns = tuple(
        float(value)
        for value in reference["peak_time_ns"].quantile(timing_quantiles)
    )

    classified = df.copy() if copy else df
    is_single_peak = classified["n_peaks"].eq(1)
    has_isolated_peak = (
        is_single_peak & classified["n_signal_like_peaks"].eq(1)
    )
    has_dominant_peak = is_single_peak & (
        classified["n_signal_like_peaks"].eq(1)
        | (
            classified["n_signal_like_peaks"].gt(1)
            & classified["largest_additional_signal_like_peak_fraction"].le(
                max_additional_peak_fraction
            ).fillna(False)
        )
    )
    has_pulse_shape = is_single_peak.copy()
    for column, bounds in shape_ranges.items():
        has_pulse_shape &= classified[column].between(*bounds).fillna(False)
    has_generic_pulse_shape = is_single_peak.copy()
    for column, bounds in generic_shape_ranges.items():
        has_generic_pulse_shape &= classified[column].between(*bounds).fillna(False)
    has_pulse_shape &= has_generic_pulse_shape
    is_trigger_aligned = (
        is_single_peak
        & classified["peak_time_ns"].between(*timing_range_ns).fillna(False)
    )
    classified["has_pulse_shape"] = has_pulse_shape.astype(bool)
    classified["has_generic_pulse_shape"] = has_generic_pulse_shape.astype(bool)
    classified["is_trigger_aligned"] = is_trigger_aligned.astype(bool)
    classified["has_isolated_peak"] = has_isolated_peak.astype(bool)
    classified["has_dominant_peak"] = has_dominant_peak.astype(bool)
    classified["signal_oscillatory"] = (
        is_single_peak
        & classified["n_signal_like_peaks"].gt(1)
        & ~has_dominant_peak
    ).astype(bool)
    classified["signal_with_small_additional_peaks"] = (
        is_single_peak
        & classified["n_signal_like_peaks"].gt(1)
        & has_dominant_peak
    ).astype(bool)
    classified["signal_bad_generic_shape"] = (
        has_dominant_peak & ~has_generic_pulse_shape
    ).astype(bool)
    classified["signal_good"] = (
        has_dominant_peak & has_pulse_shape & is_trigger_aligned
    ).astype(bool)
    classified["signal_noise_like"] = (
        has_dominant_peak & ~has_pulse_shape
    ).astype(bool)
    classified["signal_off_time"] = (
        has_dominant_peak & has_pulse_shape & ~is_trigger_aligned
    ).astype(bool)

    summary = {
        "reference_snr": float(reference_snr),
        "shape_quantiles": shape_quantiles,
        "generic_shape_ranges": generic_shape_ranges,
        "max_additional_peak_fraction": max_additional_peak_fraction,
        "timing_quantiles": timing_quantiles,
        "min_reference_pulses": int(min_reference_pulses),
        "reference_pulses": int(len(reference)),
        "shape_ranges": shape_ranges,
        "timing_range_ns": timing_range_ns,
        "counts": {
            name: int(classified[name].sum())
            for name in (
                "has_pulse_shape",
                "has_generic_pulse_shape",
                "is_trigger_aligned",
                "has_isolated_peak",
                "has_dominant_peak",
                "signal_oscillatory",
                "signal_with_small_additional_peaks",
                "signal_bad_generic_shape",
                "signal_good",
                "signal_noise_like",
                "signal_off_time",
            )
        },
    }
    return classified, summary


def build_calibration_selections(
    df,
    *,
    cut_thresholds_snr=(15.0,),
    selection_mode="standard",
    timing_reference_snr=15.0,
    peak_timing_tolerance_ns=5.0,
    timing_reference_requires_single_peak=None,
    max_allowed_peaks=6,
    include_no_peak_cuts=True,
    include_shape_cut=False,
    selection_names=None,
    min_shape_reference_pulses=100,
    shape_quantiles=(0.01, 0.99),
    materialize_selected_dfs=True,
):
    """Build the common single-voltage and batch calibration selections.

    Events with no qualifying peak, or with SNR below a configured boundary,
    are deliberately retained.  They form the pedestal population required by
    the occupancy fits.  Quality cuts are applied only above each SNR boundary.

    The returned dictionary contains the generated selection configurations,
    selected dataframes, cutflows, timing information, and learned pulse-shape
    ranges. Set ``materialize_selected_dfs=False`` to return only masks and
    cutflows, allowing callers to write selections sequentially without holding
    several full dataframe copies. Both ``Selection.ipynb`` and ``Fit.ipynb``
    use this function so the interactive and batch workflows cannot drift apart.
    """

    required_columns = {
        "snr", "n_peaks", "n_signal_like_peaks", "peak_time_ns",
        "has_dominant_peak", "has_generic_pulse_shape",
    }
    shape_columns = [
        "peak_width_ns",
        "rise_time_10_90_ns",
        "fall_time_90_10_ns",
    ]
    if include_shape_cut:
        required_columns.update(shape_columns)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Calibration selection requires missing dataframe columns: "
            f"{sorted(missing_columns)}"
        )

    cut_thresholds_snr = [float(value) for value in cut_thresholds_snr]
    if not cut_thresholds_snr:
        raise ValueError("At least one SNR selection threshold is required.")
    if any(not np.isfinite(value) for value in cut_thresholds_snr):
        raise ValueError("SNR selection thresholds must be finite.")

    selection_name_filter = None if selection_names is None else set(selection_names)
    if selection_name_filter:
        requested_modes = set()
        for requested_name in selection_name_filter:
            if requested_name == "no_peak_cuts":
                continue
            if requested_name.startswith("led_timing_above_snr"):
                requested_modes.add("timing_only")
            elif requested_name.startswith("loose_peak_timing_above_snr"):
                requested_modes.add("loose_peak_multiplicity")
            elif requested_name.startswith("pulse_quality_above_snr"):
                requested_modes.add("standard")
            elif requested_name.startswith("dark_count_quality_above_snr"):
                requested_modes.add("dark_counts")
        if len(requested_modes) > 1:
            raise ValueError(
                "selection_names mixes modes; choose names from only one "
                "selection mode."
            )
        if requested_modes:
            selection_mode = requested_modes.pop()

    if selection_mode not in VALID_CALIBRATION_SELECTION_MODES:
        raise ValueError(
            f"Unknown selection_mode={selection_mode!r}. Choose one of "
            f"{sorted(VALID_CALIBRATION_SELECTION_MODES)}."
        )
    max_allowed_peaks = int(max_allowed_peaks)
    if max_allowed_peaks < 1:
        raise ValueError("max_allowed_peaks must be at least 1.")

    if timing_reference_requires_single_peak is None:
        timing_reference_requires_single_peak = selection_mode == "standard"
    timing_reference_requires_single_peak = bool(
        timing_reference_requires_single_peak
    )

    is_single_peak = df["has_dominant_peak"].fillna(False).to_numpy(dtype=bool)
    has_generic_pulse_shape = (
        df["has_generic_pulse_shape"].fillna(False).to_numpy(dtype=bool)
    )
    is_loose_peak_multiplicity = df["n_peaks"].between(
        1, max_allowed_peaks
    ).to_numpy()
    finite_peak_time = np.isfinite(df["peak_time_ns"].to_numpy(dtype=float))
    timing_reference_mask = (
        df["snr"].ge(float(timing_reference_snr)).fillna(False).to_numpy()
        & finite_peak_time
    )
    if timing_reference_requires_single_peak:
        timing_reference_mask &= is_single_peak
        timing_reference_mask &= has_generic_pulse_shape

    timing_reference = df.loc[timing_reference_mask, "peak_time_ns"]
    uses_led_timing = selection_mode != "dark_counts"
    expected_peak_time_ns = np.nan
    allowed_peak_window_ns = None
    if uses_led_timing:
        if timing_reference.empty:
            raise ValueError(
                "No events are available to estimate the LED peak time. "
                "Lower timing_reference_snr or change the timing-reference "
                "multiplicity requirement."
            )
        timing_min = float(timing_reference.min())
        timing_max = float(timing_reference.max())
        timing_edges = np.arange(timing_min, timing_max + 1.0, 1.0)
        if timing_edges.size < 2:
            timing_edges = np.array([timing_min - 0.5, timing_min + 0.5])
        timing_counts, timing_edges = np.histogram(
            timing_reference.to_numpy(dtype=float), bins=timing_edges
        )
        peak_bin = int(np.argmax(timing_counts))
        expected_peak_time_ns = float(
            0.5 * (timing_edges[peak_bin] + timing_edges[peak_bin + 1])
        )
        tolerance = float(peak_timing_tolerance_ns)
        allowed_peak_window_ns = (
            expected_peak_time_ns - tolerance,
            expected_peak_time_ns + tolerance,
        )
        is_led_aligned = df["peak_time_ns"].between(
            *allowed_peak_window_ns
        ).to_numpy()
    else:
        is_led_aligned = np.ones(len(df), dtype=bool)

    shape_cut_ranges = {}
    shape_cut_available = False
    shape_reference = df.iloc[0:0]
    if include_shape_cut:
        finite_shapes = np.isfinite(
            df[shape_columns].to_numpy(dtype=float)
        ).all(axis=1)
        shape_reference_mask = timing_reference_mask & finite_shapes
        if uses_led_timing:
            shape_reference_mask &= is_led_aligned
        shape_reference = df.loc[shape_reference_mask, shape_columns]
        shape_cut_available = len(shape_reference) >= int(
            min_shape_reference_pulses
        )
        if shape_cut_available:
            shape_cut_ranges = {
                column: tuple(
                    float(value)
                    for value in shape_reference[column].quantile(
                        shape_quantiles
                    )
                )
                for column in shape_columns
            }

    is_pulse_shaped = np.ones(len(df), dtype=bool)
    for column, limits in shape_cut_ranges.items():
        is_pulse_shaped &= df[column].between(*limits).to_numpy()

    selection_configs = []
    if include_no_peak_cuts:
        selection_configs.append(
            {"name": "no_peak_cuts", "label": "No peak cuts", "cuts": []}
        )

    for threshold_snr in cut_thresholds_snr:
        threshold_tag = f"snr{threshold_snr:g}"
        is_at_or_below_threshold = (
            df["snr"].isna() | df["snr"].lt(threshold_snr)
        ).to_numpy()

        if selection_mode == "standard":
            selection_name = f"pulse_quality_above_{threshold_tag}"
            selection_label = "Dominant single peak + LED timing"
            quality_cuts = [
                (
                    f"SNR >= {threshold_snr:g}: dominant single peak",
                    is_at_or_below_threshold | is_single_peak,
                ),
                (
                    f"SNR >= {threshold_snr:g}: generic pulse shape",
                    is_at_or_below_threshold | has_generic_pulse_shape,
                ),
                (
                    f"SNR >= {threshold_snr:g}: LED peak timing",
                    is_at_or_below_threshold | is_led_aligned,
                ),
            ]
        elif selection_mode == "dark_counts":
            selection_name = f"dark_count_quality_above_{threshold_tag}"
            selection_label = "Dominant single peak (timing ignored)"
            quality_cuts = [
                (
                    f"SNR >= {threshold_snr:g}: dominant single peak",
                    is_at_or_below_threshold | is_single_peak,
                ),
                (
                    f"SNR >= {threshold_snr:g}: generic pulse shape",
                    is_at_or_below_threshold | has_generic_pulse_shape,
                ),
            ]
        elif selection_mode == "timing_only":
            selection_name = f"led_timing_above_{threshold_tag}"
            selection_label = "LED timing"
            quality_cuts = [
                (
                    f"SNR >= {threshold_snr:g}: LED peak timing",
                    is_at_or_below_threshold | is_led_aligned,
                )
            ]
        else:
            selection_name = f"loose_peak_timing_above_{threshold_tag}"
            selection_label = f"1-{max_allowed_peaks} peaks + LED timing"
            quality_cuts = [
                (
                    f"SNR >= {threshold_snr:g}: 1-{max_allowed_peaks} peaks",
                    is_at_or_below_threshold | is_loose_peak_multiplicity,
                ),
                (
                    f"SNR >= {threshold_snr:g}: LED peak timing",
                    is_at_or_below_threshold | is_led_aligned,
                ),
            ]

        if include_shape_cut and shape_cut_available:
            quality_cuts.append(
                (
                    f"SNR >= {threshold_snr:g}: pulse shape",
                    is_at_or_below_threshold | is_pulse_shaped,
                )
            )
        if include_shape_cut:
            shape_label = (
                " + shape" if shape_cut_available else " (shape cut skipped)"
            )
        else:
            shape_label = ""
        selection_configs.append(
            {
                "name": selection_name,
                "label": (
                    f"{selection_label}{shape_label} at SNR >= "
                    f"{threshold_snr:g}"
                ),
                "cuts": quality_cuts,
            }
        )

    if selection_name_filter is not None:
        available_names = {config["name"] for config in selection_configs}
        unknown_names = selection_name_filter.difference(available_names)
        if unknown_names:
            raise ValueError(
                f"Unknown selection names: {sorted(unknown_names)}. "
                f"Available names: {sorted(available_names)}"
            )
        selection_configs = [
            config
            for config in selection_configs
            if config["name"] in selection_name_filter
        ]
    if not selection_configs:
        raise ValueError("No calibration selection configurations are enabled.")

    materialize_selected_dfs = bool(materialize_selected_dfs)
    selected_dfs = {}
    rejected_masks = {}
    cutflows = {}
    for config in selection_configs:
        cutflow = CutFlow(len(df))
        for cut_name, keep_mask in config["cuts"]:
            cutflow.apply(cut_name, keep_mask)
        name = config["name"]
        cutflows[name] = cutflow
        if materialize_selected_dfs:
            selected_dfs[name] = df.loc[cutflow.mask].copy()
        rejected_masks[name] = ~cutflow.mask

    return {
        "selection_mode": selection_mode,
        "selection_configs": selection_configs,
        "selected_dfs": selected_dfs,
        "rejected_masks": rejected_masks,
        "cutflows": cutflows,
        "uses_led_timing": uses_led_timing,
        "timing_reference_mask": timing_reference_mask,
        "timing_reference": timing_reference,
        "timing_reference_requires_single_peak": (
            timing_reference_requires_single_peak
        ),
        "expected_peak_time_ns": expected_peak_time_ns,
        "allowed_peak_window_ns": allowed_peak_window_ns,
        "shape_reference": shape_reference,
        "shape_cut_available": shape_cut_available,
        "shape_cut_ranges": shape_cut_ranges,
    }
