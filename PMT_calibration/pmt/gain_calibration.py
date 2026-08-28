"""Utilities for combining SPE fits into a PMT high-voltage calibration."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml


_RESULT_FILENAME = re.compile(
    r"^(?P<pmt_id>.+?)_(?P<voltage_V>\d+(?:\.\d+)?)V_"
    r"(?P<acquisition>.+?)_"
    r"(?P<integration>full_waveform|led_window|fixed_pulse_window)_"
    r"(?P<selection>.+)_"
    r"(?P<model>poisson|bellamy)\.ya?ml$"
)


def load_voltage_scan_results(results_dir, pattern="*.yaml", strict=False):
    """Load new-format per-fit YAML files into one tidy dataframe.

    Parameters
    ----------
    results_dir : path-like
        Directory written by ``save_fit_results`` in ``Fit.ipynb``.
    pattern : str
        Glob selecting result files.
    strict : bool
        Raise on filenames that do not contain voltage, integration method,
        selection, and model. By default legacy filenames are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per fit. Each fitted parameter has ``<name>`` and
        ``<name>_error`` columns.
    """
    results_dir = Path(results_dir)
    rows = []
    skipped = []

    for path in sorted(results_dir.glob(pattern)):
        match = _RESULT_FILENAME.match(path.name)
        if match is None:
            skipped.append(path.name)
            continue

        with path.open() as stream:
            result = yaml.safe_load(stream)

        row = match.groupdict()
        row["voltage_V"] = float(row["voltage_V"])
        row.update({
            "source_file": str(path),
            "fit_model": result.get("fit_model"),
            "bin_width_mV_ns": result.get("bin_width"),
            "chi2": result.get("chi2"),
            "ndof": result.get("ndof"),
        })
        ndof = row["ndof"]
        row["reduced_chi2"] = (
            row["chi2"] / ndof
            if row["chi2"] is not None and ndof not in (None, 0)
            else np.nan
        )

        for name, parameter in (result.get("fit_results") or {}).items():
            value = parameter.get("value")
            lower_bound = parameter.get("lower_bound")
            upper_bound = parameter.get("upper_bound")
            row[name] = value
            row[f"{name}_error"] = parameter.get("error")
            row[f"{name}_initial"] = parameter.get("initial")
            row[f"{name}_lower_bound"] = lower_bound
            row[f"{name}_upper_bound"] = upper_bound
            row[f"{name}_fixed"] = parameter.get("fixed")
            row[f"{name}_at_bound"] = bool(
                value is not None
                and (
                    (lower_bound is not None and np.isclose(value, lower_bound))
                    or (upper_bound is not None and np.isclose(value, upper_bound))
                )
            )

        gain = result.get("gain") or {}
        row["gain"] = gain.get("value")
        row["gain_error"] = gain.get("error")
        rows.append(row)

    if strict and skipped:
        raise ValueError(
            "Unrecognized result filenames: " + ", ".join(skipped[:5])
        )
    if not rows:
        raise ValueError(
            f"No new-format SPE result files found in {results_dir}. "
            "Expected filenames containing voltage, integration, selection, and model."
        )

    summary = pd.DataFrame(rows).sort_values(
        ["pmt_id", "integration", "selection", "model", "voltage_V"]
    ).reset_index(drop=True)
    if {"sigma1_mV_ns", "q1_mV_ns"}.issubset(summary.columns):
        summary["spe_charge_resolution"] = (
            summary["sigma1_mV_ns"] / summary["q1_mV_ns"]
        )
    return summary


def select_voltage_scan(
    summary,
    *,
    model="poisson",
    integration="led_window",
    selection="pulse_quality_above_snr20",
    pmt_id=None,
    acquisition=None,
):
    """Select one internally consistent series of fits versus voltage."""
    mask = (
        summary["model"].eq(model)
        & summary["integration"].eq(integration)
        & summary["selection"].eq(selection)
    )
    if pmt_id is not None:
        mask &= summary["pmt_id"].eq(pmt_id)
    if acquisition is not None:
        mask &= summary["acquisition"].eq(acquisition)

    scan = summary.loc[mask].sort_values("voltage_V").copy()
    if scan.empty:
        raise ValueError(
            "No fits match the requested model/integration/selection filters."
        )
    duplicates = scan[scan.duplicated("voltage_V", keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "More than one fit remains at the same voltage. Specify pmt_id "
            "and/or acquisition so that the scan contains one fit per voltage."
        )
    return scan


def fit_gain_power_law(scan, reference_voltage_V=None):
    r"""Fit :math:`G(V)=G_\mathrm{ref}(V/V_\mathrm{ref})^k` in log space."""
    valid = (
        np.isfinite(scan["voltage_V"])
        & np.isfinite(scan["gain"])
        & (scan["voltage_V"] > 0)
        & (scan["gain"] > 0)
    )
    if "q1_mV_ns_at_bound" in scan.columns:
        valid &= ~scan["q1_mV_ns_at_bound"].fillna(False)
    data = scan.loc[valid]
    if len(data) < 2:
        raise ValueError("At least two valid voltages are required for a gain curve.")

    voltage = data["voltage_V"].to_numpy(float)
    gain = data["gain"].to_numpy(float)
    if reference_voltage_V is None:
        reference_voltage_V = float(np.exp(np.mean(np.log(voltage))))

    x = np.log(voltage / reference_voltage_V)
    y = np.log(gain)
    gain_error = data.get("gain_error", pd.Series(np.nan, index=data.index)).to_numpy(float)
    valid_error = np.isfinite(gain_error) & (gain_error > 0)
    weights = gain / gain_error if np.all(valid_error) else None

    design = np.column_stack([np.ones_like(x), x])
    if weights is None:
        weighted_design = design
        weighted_y = y
    else:
        weighted_design = design * weights[:, None]
        weighted_y = y * weights
    coefficients, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    log_gain_ref, exponent = coefficients

    covariance = np.full((2, 2), np.nan)
    if len(data) > 2:
        residual = y - design @ coefficients
        dof = len(data) - 2
        if weights is None:
            normal_matrix = design.T @ design
            residual_variance = np.sum(residual**2) / dof
        else:
            normal_matrix = design.T @ (weights[:, None] ** 2 * design)
            residual_variance = np.sum((weights * residual) ** 2) / dof
        covariance = residual_variance * np.linalg.inv(normal_matrix)

    gain_ref = float(np.exp(log_gain_ref))
    return {
        "reference_voltage_V": float(reference_voltage_V),
        "gain_at_reference": gain_ref,
        "gain_at_reference_error": float(gain_ref * np.sqrt(covariance[0, 0])),
        "exponent": float(exponent),
        "exponent_error": float(np.sqrt(covariance[1, 1])),
        "n_points": int(len(data)),
    }


def evaluate_gain_power_law(voltage_V, calibration):
    """Evaluate a calibration returned by :func:`fit_gain_power_law`."""
    voltage_V = np.asarray(voltage_V, dtype=float)
    return calibration["gain_at_reference"] * (
        voltage_V / calibration["reference_voltage_V"]
    ) ** calibration["exponent"]


def plot_gain_calibration(scan, *, fit_power_law=True, figsize=(12, 4.5)):
    """Plot SPE charge and gain versus PMT voltage."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    voltage = scan["voltage_V"]

    axes[0].errorbar(
        voltage, scan["q1_mV_ns"], yerr=scan.get("q1_mV_ns_error"),
        fmt="o", capsize=3,
    )
    axes[0].set(xlabel="PMT voltage [V]", ylabel=r"$Q_1$ [mV ns]")

    axes[1].errorbar(
        voltage, scan["gain"], yerr=scan.get("gain_error"),
        fmt="o", capsize=3, label="SPE fits",
    )
    if "q1_mV_ns_at_bound" in scan.columns:
        bound_hits = scan["q1_mV_ns_at_bound"].fillna(False)
        for ax, parameter in zip(axes, ["q1_mV_ns", "gain"]):
            ax.scatter(
                scan.loc[bound_hits, "voltage_V"],
                scan.loc[bound_hits, parameter],
                marker="x", s=90, color="tab:red", zorder=5,
                label="Q1 at fit bound",
            )
        if bound_hits.any():
            axes[0].legend()
    calibration = None
    usable = scan
    if "q1_mV_ns_at_bound" in scan.columns:
        usable = scan.loc[~scan["q1_mV_ns_at_bound"].fillna(False)]
    if fit_power_law and len(usable) >= 2:
        calibration = fit_gain_power_law(usable)
        voltage_curve = np.linspace(voltage.min(), voltage.max(), 300)
        axes[1].plot(
            voltage_curve,
            evaluate_gain_power_law(voltage_curve, calibration),
            label=rf"$G\propto V^{{{calibration['exponent']:.2f}}}$",
        )
        axes[1].legend()
    elif fit_power_law:
        axes[1].text(
            0.03, 0.97, "Power law not fit:\n<2 non-bound points",
            transform=axes[1].transAxes, va="top",
        )
    axes[1].set(xlabel="PMT voltage [V]", ylabel="Gain", yscale="log")
    fig.tight_layout()
    return fig, axes, calibration


def plot_fit_parameters_vs_voltage(
    scan,
    parameters=None,
    *,
    ncols=3,
    figsize_per_panel=(4.5, 3.5),
):
    """Plot fitted nuisance and response parameters against PMT voltage."""
    if parameters is None:
        preferred = [
            "q0_mV_ns", "sigma0_mV_ns", "sigma1_mV_ns",
            "spe_charge_resolution", "mu_pe", "reduced_chi2", "w", "alpha",
        ]
        parameters = [name for name in preferred if name in scan.columns]
    if not parameters:
        raise ValueError("No requested fit parameters are present in the dataframe.")

    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, parameter in zip(flat_axes, parameters):
        error_column = f"{parameter}_error"
        yerr = scan[error_column] if error_column in scan.columns else None
        ax.errorbar(
            scan["voltage_V"], scan[parameter], yerr=yerr,
            fmt="o-", capsize=3,
        )
        ax.set(xlabel="PMT voltage [V]", ylabel=parameter)
        if parameter == "reduced_chi2":
            ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    for ax in flat_axes[len(parameters):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes


def save_voltage_sweep_diagnostics_pdf(
    cache_paths,
    output_path,
    *,
    charge_methods=("full_waveform",),
    selection_names=("no_peak_cuts",),
    histogram_bins=250,
    histogram_density=True,
    histogram_log_y=True,
    histogram_range_quantiles=(0.001, 0.999),
    waveforms_per_voltage=3,
    high_snr_threshold=15.0,
    waveform_seed=12345,
):
    """Save raw charge and waveform comparisons across a voltage sweep.

    The PDF contains one overlaid charge-histogram page per requested
    integration/selection pair, followed by a high-SNR single-pulse page and a
    no-detected-peak (pedestal-like) page. No fitted functions are drawn.

    Selection and waveform preprocessing settings are read from each cache's
    versioned Selection manifest. Only charge arrays and the requested sampled
    event rows are retained in memory while the cache files are traversed.
    """

    from .io import (
        extract_pmt_voltage,
        load_baseline_reference,
        load_event_waveforms,
    )
    from .provenance import load_selection_cache_provenance
    from .selection import build_calibration_selections

    cache_paths = [Path(path) for path in cache_paths]
    if len(cache_paths) < 1:
        raise ValueError("At least one Selection cache is required")
    missing_caches = [str(path) for path in cache_paths if not path.is_file()]
    if missing_caches:
        raise FileNotFoundError(
            "Missing Selection caches: " + ", ".join(missing_caches[:5])
        )

    charge_methods = list(dict.fromkeys(str(name) for name in charge_methods))
    valid_charge_methods = {
        "full_waveform",
        "led_window",
        "fixed_pulse_window",
    }
    unknown_charge_methods = set(charge_methods).difference(valid_charge_methods)
    if unknown_charge_methods:
        raise ValueError(
            f"Unknown charge methods: {sorted(unknown_charge_methods)}"
        )
    if not charge_methods:
        raise ValueError("At least one charge method is required")

    if selection_names is not None:
        selection_names = list(dict.fromkeys(str(name) for name in selection_names))
        if not selection_names:
            raise ValueError("selection_names cannot be empty")
    histogram_bins = int(histogram_bins)
    if histogram_bins < 1:
        raise ValueError("histogram_bins must be at least 1")
    if len(histogram_range_quantiles) != 2:
        raise ValueError("histogram_range_quantiles must contain two values")
    quantile_low, quantile_high = (
        float(value) for value in histogram_range_quantiles
    )
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError(
            "histogram_range_quantiles must satisfy 0 <= low < high <= 1"
        )
    waveforms_per_voltage = int(waveforms_per_voltage)
    if waveforms_per_voltage < 0:
        raise ValueError("waveforms_per_voltage must be non-negative")
    high_snr_threshold = float(high_snr_threshold)
    if not np.isfinite(high_snr_threshold):
        raise ValueError("high_snr_threshold must be finite")

    rng = np.random.default_rng(int(waveform_seed))
    records = []
    resolved_selection_names = selection_names

    for cache_path in cache_paths:
        print(f"Loading voltage-sweep diagnostics from {cache_path}")
        dataframe = pd.read_pickle(cache_path)
        provenance = load_selection_cache_provenance(dataframe)
        preprocessing = provenance["preprocessing"]
        selection = provenance["selection"]

        generated_names = list(selection["generated_selection_names"])
        if resolved_selection_names is None:
            resolved_selection_names = generated_names
        unavailable = set(resolved_selection_names).difference(generated_names)
        if unavailable:
            raise ValueError(
                f"Cache {cache_path} does not contain configured selections "
                f"{sorted(unavailable)}; available: {generated_names}"
            )

        selection_analysis = build_calibration_selections(
            dataframe,
            cut_thresholds_snr=selection["cut_thresholds_snr"],
            selection_mode=selection["selection_mode"],
            timing_reference_snr=selection["timing_reference_snr"],
            peak_timing_tolerance_ns=selection["peak_timing_tolerance_ns"],
            timing_reference_requires_single_peak=(
                selection["timing_reference_requires_single_peak"]
            ),
            max_allowed_peaks=selection["max_allowed_peaks"],
            include_no_peak_cuts=selection["include_no_peak_cuts"],
            include_shape_cut=selection["include_shape_cut"],
            min_shape_reference_pulses=selection["min_shape_reference_pulses"],
            shape_quantiles=selection["shape_quantiles"],
            selection_names=resolved_selection_names,
        )

        method_columns = {
            "full_waveform": "area_mV_ns",
            "led_window": "charge_led_window_mV_ns",
        }
        fixed_window = preprocessing["fixed_pulse_window_charge"]
        if fixed_window.get("enabled", False):
            method_columns["fixed_pulse_window"] = fixed_window["column"]
        missing_methods = set(charge_methods).difference(method_columns)
        if missing_methods:
            raise ValueError(
                f"Cache {cache_path} has no charge data for "
                f"{sorted(missing_methods)}. Rerun Selection with those methods enabled."
            )

        charges = {}
        for selection_name in resolved_selection_names:
            selected = selection_analysis["selected_dfs"][selection_name]
            charges[selection_name] = {}
            for method in charge_methods:
                column = method_columns[method]
                values = selected[column].to_numpy(dtype=float)
                charges[selection_name][method] = values[np.isfinite(values)].copy()

        sampled_events = {}
        population_masks = {
            "high_snr": (
                dataframe["n_peaks"].eq(1)
                & dataframe["snr"].ge(high_snr_threshold)
            ),
            "pedestal": dataframe["n_peaks"].eq(0),
        }
        for population, mask in population_masks.items():
            candidates = dataframe.loc[mask]
            sample_size = min(waveforms_per_voltage, len(candidates))
            if sample_size:
                sampled_indices = rng.choice(
                    candidates.index.to_numpy(), size=sample_size, replace=False
                )
                sampled_events[population] = (
                    candidates.loc[sampled_indices].sort_index().copy()
                )
            else:
                sampled_events[population] = candidates.iloc[0:0].copy()

        records.append({
            "cache_path": cache_path,
            "voltage_V": extract_pmt_voltage(provenance["acquisition"]),
            "acquisition": provenance["acquisition"],
            "provenance": provenance,
            "charges": charges,
            "sampled_events": sampled_events,
        })
        del selection_analysis, dataframe

    records.sort(key=lambda record: record["voltage_V"])
    voltages = [record["voltage_V"] for record in records]
    if len(set(voltages)) != len(voltages):
        raise ValueError(
            f"Voltage-sweep diagnostic caches contain duplicate voltages: {voltages}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages_written = 0
    color_map = plt.get_cmap("viridis")
    colors = [
        color_map(index / max(1, len(records) - 1))
        for index in range(len(records))
    ]

    method_labels = {
        "full_waveform": "Full waveform",
        "led_window": "LED window",
        "fixed_pulse_window": "Learned fixed pulse window",
    }
    with PdfPages(output_path) as pdf:
        for method in charge_methods:
            for selection_name in resolved_selection_names:
                value_sets = [
                    record["charges"][selection_name][method]
                    for record in records
                ]
                nonempty = [values for values in value_sets if values.size]
                if not nonempty:
                    print(
                        f"Skipping empty histogram: {method}, {selection_name}"
                    )
                    continue
                range_low = min(
                    float(np.quantile(values, quantile_low))
                    for values in nonempty
                )
                range_high = max(
                    float(np.quantile(values, quantile_high))
                    for values in nonempty
                )
                if not np.isfinite(range_low) or not np.isfinite(range_high):
                    raise ValueError(
                        f"Non-finite histogram range for {method}/{selection_name}"
                    )
                if range_low == range_high:
                    padding = max(1.0, abs(range_low) * 0.01)
                    range_low -= padding
                    range_high += padding
                edges = np.linspace(range_low, range_high, histogram_bins + 1)

                fig, ax = plt.subplots(figsize=(11.5, 7.2))
                for record, values, color in zip(records, value_sets, colors):
                    counts, _ = np.histogram(values, bins=edges)
                    if histogram_density and counts.sum():
                        counts = counts / (counts.sum() * np.diff(edges))
                    ax.stairs(
                        counts,
                        edges,
                        color=color,
                        linewidth=1.5,
                        label=(
                            f"{record['voltage_V']:g} V "
                            f"(N={len(values):,})"
                        ),
                    )
                if histogram_log_y:
                    ax.set_yscale("log")
                ax.set(
                    xlabel="Charge [mV ns]",
                    ylabel=("Density" if histogram_density else "Events"),
                    title=(
                        f"Raw charge distributions across voltage\n"
                        f"{method_labels[method]} | {selection_name}"
                    ),
                )
                ax.legend(title="PMT voltage", fontsize=9, ncols=2)
                ax.grid(True, alpha=0.3)
                ax.text(
                    0.01,
                    0.01,
                    (
                        f"Common display range: per-voltage "
                        f"{100 * quantile_low:g}–{100 * quantile_high:g}% quantiles; "
                        "no fitted functions"
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                pages_written += 1

        if waveforms_per_voltage > 0:
            waveform_pages = [
                (
                    "high_snr",
                    f"High-SNR single-peak waveforms (SNR >= {high_snr_threshold:g})",
                ),
                ("pedestal", "Pedestal-like waveforms (no detected peak)"),
            ]
            for population, title in waveform_pages:
                fig, ax = plt.subplots(figsize=(12.5, 7.2))
                plotted = 0
                for record, color in zip(records, colors):
                    events = record["sampled_events"][population].copy()
                    if events.empty:
                        print(
                            f"No {population} waveform candidates at "
                            f"{record['voltage_V']:g} V"
                        )
                        continue

                    provenance = record["provenance"]
                    source_by_name = {
                        Path(path).name: Path(path)
                        for path in provenance["source_files"]
                    }
                    event_names = {
                        Path(path).name for path in events["event_file"].unique()
                    }
                    missing_sources = sorted(
                        name
                        for name in event_names
                        if name not in source_by_name
                        or not source_by_name[name].is_file()
                    )
                    if missing_sources:
                        raise FileNotFoundError(
                            f"Missing raw waveform sources for {record['acquisition']}: "
                            f"{missing_sources}"
                        )
                    events["event_file"] = events["event_file"].map(
                        lambda path: str(source_by_name[Path(path).name])
                    )

                    preprocessing = provenance["preprocessing"]
                    reference_path_value = preprocessing["baseline_reference_path"]
                    reference_time_ns = None
                    reference_mV = None
                    if reference_path_value is not None:
                        reference_path = Path(reference_path_value)
                        if not reference_path.is_file():
                            raise FileNotFoundError(
                                f"Missing baseline reference: {reference_path}"
                            )
                        reference = load_baseline_reference(reference_path)
                        reference_time_ns = reference["time_ns"]
                        reference_mV = reference["baseline_template_mV"]

                    time_ns, waveforms_mV = load_event_waveforms(
                        events,
                        channel=preprocessing["channel"],
                        baseline_window_ns=tuple(
                            preprocessing["baseline_window_ns"]
                        ),
                        baseline_reference_time_ns=reference_time_ns,
                        baseline_reference_mV=reference_mV,
                    )
                    for waveform_index, waveform in enumerate(waveforms_mV):
                        ax.plot(
                            time_ns,
                            waveform,
                            color=color,
                            linewidth=1.0,
                            alpha=0.72,
                            label=(
                                f"{record['voltage_V']:g} V "
                                f"(N={len(waveforms_mV)})"
                                if waveform_index == 0
                                else None
                            ),
                        )
                    plotted += len(waveforms_mV)

                if plotted:
                    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
                    ax.set(
                        xlabel="Time [ns]",
                        ylabel="Voltage [mV]",
                        title=(
                            f"{title}\n"
                            f"up to {waveforms_per_voltage} sampled per voltage"
                        ),
                    )
                    ax.legend(title="PMT voltage", fontsize=9, ncols=2)
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    pdf.savefig(fig, bbox_inches="tight")
                    pages_written += 1
                plt.close(fig)

    print(
        f"Saved {pages_written}-page voltage-sweep diagnostic PDF to "
        f"{output_path}"
    )
    return {
        "output_path": output_path,
        "pages": pages_written,
        "voltages_V": voltages,
        "charge_methods": tuple(charge_methods),
        "selection_names": tuple(resolved_selection_names),
        "waveforms_per_voltage": waveforms_per_voltage,
    }


__all__ = [
    "load_voltage_scan_results",
    "select_voltage_scan",
    "fit_gain_power_law",
    "evaluate_gain_power_law",
    "plot_gain_calibration",
    "plot_fit_parameters_vs_voltage",
    "save_voltage_sweep_diagnostics_pdf",
]
