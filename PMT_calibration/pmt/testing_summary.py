"""Generate multipage PDFs from the diagnostics in ``testing.ipynb``."""

from __future__ import annotations

from contextlib import contextmanager
import gc
import pickle
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import nbformat
import numpy as np
import pandas as pd

from .io import baseline_subtraction, extract_pmt_voltage, load_event_waveforms


SHAPE_DIAGNOSTICS_CACHE_VERSION = 3


def save_shape_diagnostics_cache(path, payload):
    """Atomically save reduced shape metrics and event locators."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = dict(payload)
    cached["cache_version"] = SHAPE_DIAGNOSTICS_CACHE_VERSION
    temporary_path = path.with_suffix(path.suffix + ".partial")
    with temporary_path.open("wb") as stream:
        pickle.dump(cached, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)
    return path


def load_shape_diagnostics_cache(path):
    """Load and validate a reduced shape-diagnostics artifact."""

    path = Path(path)
    with path.open("rb") as stream:
        cached = pickle.load(stream)
    if cached.get("cache_version") != SHAPE_DIAGNOSTICS_CACHE_VERSION:
        raise ValueError(
            f"Unsupported shape cache version in {path}: "
            f"{cached.get('cache_version')!r}; run the first Testing cell once "
            "to refresh the reduced cache."
        )
    source_cache_path = Path(cached["source_cache_path"])
    if source_cache_path.is_file():
        current_mtime = source_cache_path.stat().st_mtime_ns
        if current_mtime != int(cached["source_cache_mtime_ns"]):
            warnings.warn(
                f"Shape cache {path} predates changes to {source_cache_path}; "
                "rerun the main Testing cell for fully current metrics.",
                stacklevel=2,
            )
    return cached


def generate_shape_diagnostics_pdf_from_cache(
    cache_path,
    output_path,
    *,
    signal_reject_upper_fraction=0.05,
    pedestal_growth_ratio_max=2.0,
    pedestal_rms_variation_max=0.20,
    worst_waveforms_to_plot=10,
    metric_histogram_bins=60,
    metric_qdc_bins=100,
    qdc_histogram_bins=40,
    fixed_qdc_range_mV_ns=None,
    fixed_qdc_waveforms_to_plot=0,
    fixed_qdc_population="both",
):
    """Regenerate the shape PDF without a full raw-waveform scan."""

    cached = load_shape_diagnostics_cache(cache_path)
    signal_reject_upper_fraction = float(signal_reject_upper_fraction)
    pedestal_growth_ratio_max = float(pedestal_growth_ratio_max)
    pedestal_rms_variation_max = float(pedestal_rms_variation_max)
    worst_waveforms_to_plot = int(worst_waveforms_to_plot)
    if not 0.0 <= signal_reject_upper_fraction < 1.0:
        raise ValueError("signal_reject_upper_fraction must be in [0, 1)")
    if pedestal_growth_ratio_max <= 0.0:
        raise ValueError("pedestal_growth_ratio_max must be positive")
    if pedestal_rms_variation_max < 0.0:
        raise ValueError("pedestal_rms_variation_max must be non-negative")
    if worst_waveforms_to_plot < 0:
        raise ValueError("worst_waveforms_to_plot must be non-negative")
    fixed_qdc_waveforms_to_plot = int(fixed_qdc_waveforms_to_plot)
    if fixed_qdc_waveforms_to_plot < 0:
        raise ValueError("fixed_qdc_waveforms_to_plot must be non-negative")
    if fixed_qdc_population not in {"both", "pedestal", "signal"}:
        raise ValueError("fixed_qdc_population must be 'both', 'pedestal', or 'signal'")
    if fixed_qdc_range_mV_ns is not None:
        if len(fixed_qdc_range_mV_ns) != 2:
            raise ValueError("fixed_qdc_range_mV_ns must contain two bounds")
        fixed_qdc_low, fixed_qdc_high = map(float, fixed_qdc_range_mV_ns)
        if not np.isfinite([fixed_qdc_low, fixed_qdc_high]).all():
            raise ValueError("fixed QDC range bounds must be finite")
        if fixed_qdc_low >= fixed_qdc_high:
            raise ValueError("fixed QDC range lower bound must be below upper bound")
    elif fixed_qdc_waveforms_to_plot:
        raise ValueError(
            "fixed_qdc_range_mV_ns is required when "
            "fixed_qdc_waveforms_to_plot is nonzero"
        )

    mismatch = np.asarray(cached["signal_template_mismatch"], dtype=float)
    growth = np.asarray(cached["pedestal_noise_growth_ratio"], dtype=float)
    rms_variation = np.asarray(
        cached["pedestal_local_rms_relative_span"], dtype=float
    )
    signal_qdc = np.asarray(cached["signal_qdc_mV_ns"], dtype=float)
    pedestal_qdc = np.asarray(cached["pedestal_qdc_mV_ns"], dtype=float)
    template_time = np.asarray(cached["template_relative_time_ns"], dtype=float)
    template = np.asarray(cached["signal_shape_template"], dtype=float)
    sample_time = np.asarray(cached["sample_time_ns"], dtype=float)
    signal_events = cached["signal_events"].copy()
    pedestal_events = cached["pedestal_events"].copy()
    pedestal_event_weight_per_s = cached.get("pedestal_event_weight_per_s")
    signal_event_weight_per_s = cached.get("signal_event_weight_per_s")
    qdc_uses_rate = (
        pedestal_event_weight_per_s is not None
        and signal_event_weight_per_s is not None
    )
    finite_mismatch = np.isfinite(mismatch)
    finite_growth = np.isfinite(growth)
    finite_rms_variation = np.isfinite(rms_variation)
    mismatch_threshold = (
        float(np.quantile(
            mismatch[finite_mismatch], 1.0 - signal_reject_upper_fraction
        ))
        if np.any(finite_mismatch) else np.nan
    )
    signal_keep = finite_mismatch & (mismatch <= mismatch_threshold)
    pedestal_growth_keep = finite_growth & (growth <= pedestal_growth_ratio_max)
    pedestal_rms_keep = (
        finite_rms_variation & (rms_variation <= pedestal_rms_variation_max)
    )
    pedestal_keep = pedestal_growth_keep & pedestal_rms_keep

    def top_indices(values, mask, count):
        indices = np.flatnonzero(mask)
        if not len(indices) or count == 0:
            return np.asarray([], dtype=int)
        order = np.argsort(values[indices])[::-1]
        return indices[order[:count]]

    signal_rejected_indices = top_indices(
        mismatch, finite_mismatch & ~signal_keep, worst_waveforms_to_plot
    )
    signal_boundary_indices = top_indices(
        mismatch, signal_keep, worst_waveforms_to_plot
    )
    pedestal_rejected_indices = top_indices(
        growth, finite_growth & ~pedestal_growth_keep, worst_waveforms_to_plot
    )
    pedestal_boundary_indices = top_indices(
        growth, pedestal_growth_keep, worst_waveforms_to_plot
    )
    pedestal_rms_rejected_indices = top_indices(
        rms_variation,
        finite_rms_variation & ~pedestal_rms_keep,
        worst_waveforms_to_plot,
    )
    pedestal_rms_boundary_indices = top_indices(
        rms_variation, pedestal_rms_keep, worst_waveforms_to_plot
    )

    preprocessing = cached["preprocessing"]
    reference_time = cached.get("baseline_reference_time_ns")
    reference_mV = cached.get("baseline_reference_mV")

    fixed_qdc_records = []
    if fixed_qdc_waveforms_to_plot:
        required_event_columns = {
            "fixed_qdc_mV_ns",
            "event_specific_qdc_mV_ns",
            "raw_full_trace_qdc_mV_ns",
        }
        missing_by_population = {
            population: sorted(required_event_columns.difference(events.columns))
            for population, events in (
                ("signal", signal_events), ("pedestal", pedestal_events)
            )
            if population in {fixed_qdc_population, "signal", "pedestal"}
            and (
                fixed_qdc_population == "both"
                or population == fixed_qdc_population
            )
            and required_event_columns.difference(events.columns)
        }
        if missing_by_population:
            raise ValueError(
                "Shape cache lacks fixed-QDC comparison fields "
                f"{missing_by_population}; rerun the first Testing cell once."
            )

        def eligible_events(events, population):
            frame = events.copy()
            fixed_values = frame["fixed_qdc_mV_ns"].to_numpy(dtype=float)
            in_range = (
                np.isfinite(fixed_values)
                & (fixed_values >= fixed_qdc_low)
                & (fixed_values <= fixed_qdc_high)
            )
            frame = frame.loc[in_range].copy()
            frame["_population"] = population
            return frame.sort_values("fixed_qdc_mV_ns").reset_index(drop=True)

        eligible = {
            "signal": eligible_events(signal_events, "signal"),
            "pedestal": eligible_events(pedestal_events, "pedestal"),
        }

        def evenly_spaced_rows(frame, count):
            count = min(int(count), len(frame))
            if count == 0:
                return frame.iloc[:0].copy()
            positions = np.linspace(0, len(frame) - 1, count)
            positions = np.rint(positions).astype(int)
            return frame.iloc[positions].copy()

        requested = fixed_qdc_waveforms_to_plot
        if fixed_qdc_population == "both":
            allocations = {
                "signal": min(len(eligible["signal"]), (requested + 1) // 2),
                "pedestal": min(len(eligible["pedestal"]), requested // 2),
            }
            remaining = requested - sum(allocations.values())
            for population in ("signal", "pedestal"):
                extra_capacity = len(eligible[population]) - allocations[population]
                extra = min(remaining, extra_capacity)
                allocations[population] += extra
                remaining -= extra
            selected_events = pd.concat(
                [
                    evenly_spaced_rows(eligible[population], allocations[population])
                    for population in ("signal", "pedestal")
                ],
                ignore_index=True,
            )
        else:
            selected_events = evenly_spaced_rows(
                eligible[fixed_qdc_population], requested
            ).reset_index(drop=True)
        selected_events = selected_events.sort_values(
            ["fixed_qdc_mV_ns", "_population"]
        ).reset_index(drop=True)

        if len(selected_events):
            fixed_window_ns = tuple(cached["fixed_window_ns"])
            raw_time_ns, raw_waveforms = load_event_waveforms(
                selected_events,
                channel=preprocessing["channel"],
                baseline_window_ns=tuple(preprocessing["baseline_window_ns"]),
                baseline_reference_time_ns=reference_time,
                baseline_reference_mV=reference_mV,
                subtract_baseline=False,
            )
            fixed_waveforms, _ = baseline_subtraction(
                raw_time_ns,
                raw_waveforms,
                baseline_window_ns=tuple(preprocessing["baseline_window_ns"]),
                baseline_reference_time_ns=reference_time,
                baseline_reference_mV=reference_mV,
            )
            for (_, event), raw_waveform, fixed_waveform in zip(
                selected_events.iterrows(), raw_waveforms, fixed_waveforms
            ):
                population = str(event["_population"])
                if population == "signal":
                    event_waveform = fixed_waveform
                    peak_time_ns = float(event["peak_time_ns"])
                    peak_width_ns = float(event["peak_width_ns"])
                    event_window_ns = (
                        peak_time_ns - peak_width_ns,
                        peak_time_ns + 2.0 * peak_width_ns,
                    )
                    event_baseline_label = "baseline-window subtracted"
                else:
                    if reference_mV is None:
                        event_waveform = np.asarray(raw_waveform, dtype=float).copy()
                    else:
                        event_waveform = (
                            np.asarray(raw_waveform, dtype=float)
                            - np.asarray(reference_mV, dtype=float)
                        )
                    event_waveform -= np.mean(event_waveform)
                    event_window_ns = (0.0, 80.0)
                    event_baseline_label = "full-trace mean subtracted"
                fixed_qdc_records.append({
                    "population": population,
                    "time_ns": raw_time_ns,
                    "fixed_waveform_mV": -np.asarray(fixed_waveform, dtype=float),
                    "event_waveform_mV": -np.asarray(event_waveform, dtype=float),
                    "raw_waveform_mV": -np.asarray(raw_waveform, dtype=float),
                    "fixed_window_ns": fixed_window_ns,
                    "event_window_ns": event_window_ns,
                    "fixed_qdc_mV_ns": float(event["fixed_qdc_mV_ns"]),
                    "event_qdc_mV_ns": float(event["event_specific_qdc_mV_ns"]),
                    "raw_qdc_mV_ns": float(event["raw_full_trace_qdc_mV_ns"]),
                    "event_baseline_label": event_baseline_label,
                })
        else:
            print(
                "No cached events found in fixed-window QDC range "
                f"[{fixed_qdc_low:g}, {fixed_qdc_high:g}] mV ns for "
                f"population={fixed_qdc_population!r}."
            )

    def load_signal_records(indices):
        if not len(indices):
            return []
        events = signal_events.iloc[indices]
        time_ns, waveforms = load_event_waveforms(
            events,
            channel=preprocessing["channel"],
            baseline_window_ns=tuple(preprocessing["baseline_window_ns"]),
            baseline_reference_time_ns=reference_time,
            baseline_reference_mV=reference_mV,
        )
        records = []
        for index, (_, event), waveform in zip(indices, events.iterrows(), waveforms):
            sample_times = float(event["peak_time_ns"]) + template_time
            if sample_times[0] < time_ns[0] or sample_times[-1] > time_ns[-1]:
                continue
            aligned = np.interp(sample_times, time_ns, -waveform)
            amplitude = float(np.max(aligned))
            if amplitude <= 0.0:
                continue
            records.append({
                "metric": float(mismatch[index]),
                "qdc_mV_ns": float(signal_qdc[index]),
                "normalized_shape": aligned / amplitude,
            })
        return records

    def load_pedestal_records(indices, metric_values, metric_name):
        if not len(indices):
            return []
        events = pedestal_events.iloc[indices]
        _, waveforms = load_event_waveforms(
            events,
            channel=preprocessing["channel"],
            baseline_window_ns=tuple(preprocessing["baseline_window_ns"]),
            baseline_reference_time_ns=reference_time,
            baseline_reference_mV=reference_mV,
        )
        return [
            {
                "metric": float(metric_values[index]),
                "metric_name": metric_name,
                "qdc_mV_ns": float(pedestal_qdc[index]),
                "waveform_mV": waveform - np.median(waveform),
            }
            for index, waveform in zip(indices, waveforms)
        ]

    signal_rejected_records = load_signal_records(signal_rejected_indices)
    signal_boundary_records = load_signal_records(signal_boundary_indices)
    pedestal_rejected_records = load_pedestal_records(
        pedestal_rejected_indices, growth, "growth"
    )
    pedestal_boundary_records = load_pedestal_records(
        pedestal_boundary_indices, growth, "growth"
    )
    pedestal_rms_rejected_records = load_pedestal_records(
        pedestal_rms_rejected_indices, rms_variation, "local RMS span"
    )
    pedestal_rms_boundary_records = load_pedestal_records(
        pedestal_rms_boundary_indices, rms_variation, "local RMS span"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    pages = 0
    with PdfPages(temporary_path) as pdf:
        def save(figure):
            nonlocal pages
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages += 1

        paired_mismatch = finite_mismatch & np.isfinite(signal_qdc)
        fig, axes_grid = plt.subplots(2, 2, figsize=(16, 11))
        axes = axes_grid.ravel()
        axes[0].plot(template_time, template, color="black", linewidth=2.0)
        axes[0].set(
            xlabel="Time relative to peak [ns]",
            ylabel="Normalized pulse amplitude",
            title=(
                "Median high-SNR signal template "
                f"(N={cached['n_template_references']})"
            ),
        )
        if np.any(finite_mismatch):
            finite_values = mismatch[finite_mismatch]
            axes[1].hist(
                finite_values, bins=metric_histogram_bins,
                histtype="step", linewidth=2.0, color="tab:blue",
            )
            axes[1].axvline(
                mismatch_threshold, color="tab:red", linestyle="--",
                label=f"reject above {mismatch_threshold:.4g}",
            )
            sorted_values = np.sort(finite_values)
            cdf_percent = 100.0 * np.arange(1, len(sorted_values) + 1) / len(
                sorted_values
            )
            kept_percent = 100.0 * np.mean(sorted_values <= mismatch_threshold)
            axes[2].step(
                sorted_values, cdf_percent, where="post",
                color="tab:blue", linewidth=1.8,
            )
            axes[2].axvline(
                mismatch_threshold, color="tab:red", linestyle="--",
                label=(
                    f"cut={mismatch_threshold:.4g}: {kept_percent:.2f}% kept, "
                    f"{100.0 - kept_percent:.2f}% rejected"
                ),
            )
            axes[2].axhline(
                kept_percent, color="tab:red", linestyle=":", alpha=0.7
            )
        axes[1].set(
            xlabel="Normalized template mismatch", ylabel="Blue events",
            title="Blue signal template-mismatch distribution",
        )
        axes[1].legend(fontsize=8)
        axes[2].set(
            xlabel="Normalized template mismatch",
            ylabel="Events with mismatch ≤ x [%]",
            title="Signal mismatch empirical CDF", ylim=(0.0, 100.0),
        )
        axes[2].legend(fontsize=8)
        if np.any(paired_mismatch):
            image = axes[3].hist2d(
                mismatch[paired_mismatch], signal_qdc[paired_mismatch],
                bins=metric_qdc_bins, cmap="jet", norm=LogNorm(),
            )
            fig.colorbar(image[3], ax=axes[3], label="Waveforms per bin")
        axes[3].set(
            xlabel="Normalized template mismatch",
            ylabel="Event-by-event QDC [mV ns]",
            title="Blue signal mismatch vs QDC",
        )
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        save(fig)

        def plot_signal_examples(records, title, color):
            if not records:
                return
            ncols = 2
            nrows = int(np.ceil(len(records) / ncols))
            figure, axes = plt.subplots(
                nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False,
                sharex=True, sharey=True,
            )
            for ax, record in zip(axes.ravel(), records):
                ax.plot(
                    template_time, template, color="black", linewidth=1.8,
                    label="median template",
                )
                ax.plot(
                    template_time, record["normalized_shape"], color=color,
                    linewidth=1.0, label="waveform",
                )
                ax.set_title(
                    f"mismatch={record['metric']:.3g}; "
                    f"QDC={record['qdc_mV_ns']:.3g} mV ns", fontsize=9,
                )
                ax.set(
                    xlabel="Time relative to peak [ns]",
                    ylabel="Normalized amplitude",
                )
                ax.grid(True, alpha=0.3)
            for ax in axes.ravel()[len(records):]:
                ax.set_visible(False)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center")
            figure.suptitle(title)
            figure.tight_layout(rect=(0, 0.04, 1, 0.96))
            save(figure)

        plot_signal_examples(
            signal_rejected_records,
            f"Rejected blue signals: top {100 * signal_reject_upper_fraction:g}% "
            f"template mismatch (threshold {mismatch_threshold:.4g})",
            "tab:blue",
        )
        plot_signal_examples(
            signal_boundary_records,
            "Highest-mismatch blue signals still accepted after removing "
            f"the top {100 * signal_reject_upper_fraction:g}% "
            f"(threshold {mismatch_threshold:.4g})",
            "tab:cyan",
        )

        paired_growth = finite_growth & np.isfinite(pedestal_qdc)
        if np.any(finite_growth):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].hist(
                growth[finite_growth], bins=metric_histogram_bins,
                histtype="step", linewidth=2.0, color="tab:orange",
            )
            axes[0].axvline(1.0, color="black", linestyle="--")
            axes[0].axvline(
                pedestal_growth_ratio_max, color="tab:red", linestyle="--",
                label=f"reject above {pedestal_growth_ratio_max:g}",
            )
            axes[0].set(
                xlabel="Late/early robust-RMS growth ratio",
                ylabel="Pedestal events",
                title="Pedestal noise-growth distribution",
            )
            axes[0].legend(fontsize=8)
            if np.any(paired_growth):
                image = axes[1].hist2d(
                    growth[paired_growth], pedestal_qdc[paired_growth],
                    bins=metric_qdc_bins, cmap="jet", norm=LogNorm(),
                )
                fig.colorbar(image[3], ax=axes[1], label="Waveforms per bin")
            axes[1].set(
                xlabel="Late/early robust-RMS growth ratio",
                ylabel="Pedestal 0–80 ns QDC [mV ns]",
                title="Pedestal noise growth vs QDC",
            )
            for ax in axes:
                ax.grid(True, which="both", alpha=0.3)
            fig.tight_layout()
            save(fig)

        paired_rms_variation = finite_rms_variation & np.isfinite(pedestal_qdc)
        if np.any(finite_rms_variation):
            finite_values = rms_variation[finite_rms_variation]
            rms_survival_percent = 100.0 * np.mean(
                finite_values <= pedestal_rms_variation_max
            )
            sorted_values = np.sort(finite_values)
            cdf_percent = (
                100.0 * np.arange(1, len(sorted_values) + 1)
                / len(sorted_values)
            )
            fig, axes = plt.subplots(1, 3, figsize=(19, 6))
            axes[0].hist(
                finite_values, bins=metric_histogram_bins,
                histtype="step", linewidth=2.0, color="tab:brown",
            )
            axes[0].axvline(
                pedestal_rms_variation_max, color="tab:red", linestyle="--",
                label=(
                    f"cut={pedestal_rms_variation_max:.1%}; "
                    f"survival={rms_survival_percent:.2f}%"
                ),
            )
            axes[0].set(
                xlabel="Local RMS relative span (P90−P10)/median",
                ylabel="Pedestal events",
                title="Pedestal local-RMS constancy",
            )
            axes[0].legend(fontsize=8)
            axes[1].step(
                sorted_values, cdf_percent, where="post",
                color="tab:brown", linewidth=1.8,
            )
            axes[1].axvline(
                pedestal_rms_variation_max, color="tab:red", linestyle="--",
                label=f"{rms_survival_percent:.2f}% retained",
            )
            axes[1].axhline(
                rms_survival_percent, color="tab:red", linestyle=":", alpha=0.7,
            )
            axes[1].set(
                xlabel="Local RMS relative span (P90−P10)/median",
                ylabel="Events with span ≤ x [%]",
                title="Pedestal local-RMS span CDF", ylim=(0.0, 100.0),
            )
            axes[1].legend(fontsize=8)
            if np.any(paired_rms_variation):
                image = axes[2].hist2d(
                    rms_variation[paired_rms_variation],
                    pedestal_qdc[paired_rms_variation],
                    bins=metric_qdc_bins, cmap="jet", norm=LogNorm(),
                )
                fig.colorbar(image[3], ax=axes[2], label="Waveforms per bin")
            axes[2].set(
                xlabel="Local RMS relative span (P90−P10)/median",
                ylabel="Pedestal 0–80 ns QDC [mV ns]",
                title="Pedestal local-RMS variation vs QDC",
            )
            for ax in axes:
                ax.grid(True, which="both", alpha=0.3)
            fig.suptitle(
                f"Robust RMS in {cached['pedestal_local_rms_window_ns']:g} ns "
                "windows across the full trace"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            save(fig)

        before_pedestal_qdc = pedestal_qdc[np.isfinite(pedestal_qdc)]
        before_signal_qdc = signal_qdc[np.isfinite(signal_qdc)]
        before_combined_qdc = np.concatenate([
            before_pedestal_qdc, before_signal_qdc
        ])
        kept_pedestal_qdc = pedestal_qdc[
            pedestal_keep & np.isfinite(pedestal_qdc)
        ]
        kept_signal_qdc = signal_qdc[signal_keep & np.isfinite(signal_qdc)]
        kept_combined_qdc = np.concatenate([
            kept_pedestal_qdc, kept_signal_qdc
        ])
        if len(before_combined_qdc):
            def survival_percent(count, total):
                return 100.0 * count / total if total else np.nan

            def population_weights(values, population):
                if not qdc_uses_rate:
                    return None
                event_weight = {
                    "pedestal": pedestal_event_weight_per_s,
                    "signal": signal_event_weight_per_s,
                }[population]
                return np.full(len(values), event_weight, dtype=float)

            def combined_weights(pedestal_values, signal_values):
                if not qdc_uses_rate:
                    return None
                return np.concatenate([
                    population_weights(pedestal_values, "pedestal"),
                    population_weights(signal_values, "signal"),
                ])

            final_pedestal_survival = survival_percent(
                len(kept_pedestal_qdc), len(before_pedestal_qdc)
            )
            final_signal_survival = survival_percent(
                len(kept_signal_qdc), len(before_signal_qdc)
            )
            low = float(np.min(before_combined_qdc))
            high = float(np.max(before_combined_qdc))
            if low == high:
                low -= 0.5
                high += 0.5
            edges = np.linspace(low, high, int(qdc_histogram_bins) + 1)
            fig, axes_grid = plt.subplots(
                4, 2, figsize=(16, 20), sharex=True, sharey=True
            )
            qdc_states = (
                ("Before either cut", before_pedestal_qdc, before_signal_qdc),
                ("After mismatch cut only", before_pedestal_qdc, kept_signal_qdc),
                ("After pedestal cuts only", kept_pedestal_qdc, before_signal_qdc),
                ("After both cuts", kept_pedestal_qdc, kept_signal_qdc),
            )
            for row, (state, pedestal_values, signal_values) in enumerate(qdc_states):
                separate_ax = axes_grid[row, 0]
                combined_ax = axes_grid[row, 1]
                pedestal_survival = survival_percent(
                    len(pedestal_values), len(before_pedestal_qdc)
                )
                signal_survival = survival_percent(
                    len(signal_values), len(before_signal_qdc)
                )
                separate_ax.hist(
                    pedestal_values, bins=edges, histtype="step",
                    linewidth=2.0, color="tab:orange", alpha=0.7,
                    weights=population_weights(pedestal_values, "pedestal"),
                    label=(
                        f"pedestal (N={len(pedestal_values):,}; "
                        f"survival={pedestal_survival:.2f}%)"
                    ),
                )
                separate_ax.hist(
                    signal_values, bins=edges, histtype="step",
                    linewidth=2.0, color="tab:blue", alpha=0.7,
                    weights=population_weights(signal_values, "signal"),
                    label=(
                        f"signal (N={len(signal_values):,}; "
                        f"survival={signal_survival:.2f}%)"
                    ),
                )
                separate_ax.set_title(f"{state} — separate populations")
                combined_values = np.concatenate([pedestal_values, signal_values])
                combined_survival = survival_percent(
                    len(combined_values), len(before_combined_qdc)
                )
                combined_ax.hist(
                    combined_values, bins=edges, histtype="step", linewidth=2.2,
                    color="tab:purple", alpha=0.7,
                    weights=combined_weights(pedestal_values, signal_values),
                    label=(
                        f"combined (N={len(combined_values):,}; "
                        f"survival={combined_survival:.2f}%)"
                    ),
                )
                combined_ax.set_title(f"{state} — combined population")
            for ax in axes_grid.ravel():
                ax.set(
                    xlabel="Event-by-event QDC [mV ns]",
                    ylabel=(
                        "Waveforms / s / bin" if qdc_uses_rate else "Waveforms"
                    ),
                    yscale="log",
                )
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=8)
            fig.suptitle(
                f"QDC before/after shape cuts: reject top "
                f"{100 * signal_reject_upper_fraction:g}% signal mismatch; "
                f"pedestal growth > {pedestal_growth_ratio_max:g} or local RMS "
                f"span > {pedestal_rms_variation_max:.1%}\n"
                f"Final cut survival — signal: {final_signal_survival:.2f}%; "
                f"pedestal: {final_pedestal_survival:.2f}%"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            save(fig)

        if fixed_qdc_records:
            method_colors = ("tab:purple", "tab:blue", "tab:green")
            figure, axes = plt.subplots(
                len(fixed_qdc_records), 3,
                figsize=(18, 3.4 * len(fixed_qdc_records)),
                squeeze=False, sharex=True,
            )

            def format_change(value, fixed_value):
                change = value - fixed_value
                if fixed_value == 0.0:
                    return f"QDC={value:.4g}; Δ={change:+.4g} mV ns"
                return (
                    f"QDC={value:.4g}; Δ={change:+.4g} mV ns "
                    f"({value / fixed_value:.3g}× fixed)"
                )

            for row, record in enumerate(fixed_qdc_records):
                time_ns = record["time_ns"]
                panels = (
                    (
                        record["fixed_waveform_mV"],
                        record["fixed_window_ns"],
                        "Fixed learned window "
                        f"[{record['fixed_window_ns'][0]:.3g}, "
                        f"{record['fixed_window_ns'][1]:.3g}] ns\n"
                        f"QDC={record['fixed_qdc_mV_ns']:.4g} mV ns; "
                        "baseline-window subtracted",
                    ),
                    (
                        record["event_waveform_mV"],
                        record["event_window_ns"],
                        "Event-specific window "
                        f"[{record['event_window_ns'][0]:.3g}, "
                        f"{record['event_window_ns'][1]:.3g}] ns\n"
                        + format_change(
                            record["event_qdc_mV_ns"],
                            record["fixed_qdc_mV_ns"],
                        )
                        + f"; {record['event_baseline_label']}",
                    ),
                    (
                        record["raw_waveform_mV"],
                        (float(time_ns[0]), float(time_ns[-1])),
                        "Raw full trace "
                        f"[{time_ns[0]:.3g}, {time_ns[-1]:.3g}] ns "
                        "(no baseline subtraction)\n"
                        + format_change(
                            record["raw_qdc_mV_ns"],
                            record["fixed_qdc_mV_ns"],
                        ),
                    ),
                )
                row_values = np.concatenate([panel[0] for panel in panels])
                finite_row_values = row_values[np.isfinite(row_values)]
                if len(finite_row_values):
                    row_low, row_high = np.min(finite_row_values), np.max(
                        finite_row_values
                    )
                    row_margin = max(0.05 * (row_high - row_low), 1e-6)
                for column, (waveform, window_ns, title) in enumerate(panels):
                    ax = axes[row, column]
                    color = method_colors[column]
                    start_ns, stop_ns = window_ns
                    integration_mask = (
                        (time_ns >= start_ns) & (time_ns <= stop_ns)
                    )
                    ax.plot(time_ns, waveform, color=color, linewidth=1.0)
                    ax.fill_between(
                        time_ns, 0.0, waveform, where=integration_mask,
                        color=color, alpha=0.18, interpolate=True,
                        label="integrated region",
                    )
                    ax.axvline(start_ns, color=color, linestyle="--", alpha=0.8)
                    ax.axvline(stop_ns, color=color, linestyle="--", alpha=0.8)
                    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
                    ax.set_title(
                        f"{record['population']} | {title}", fontsize=8.5
                    )
                    ax.set(xlabel="Time [ns]", ylabel="Pulse-positive voltage [mV]")
                    if len(finite_row_values):
                        ax.set_ylim(row_low - row_margin, row_high + row_margin)
                    ax.grid(True, which="both", alpha=0.3)
            figure.suptitle(
                "Same events under three QDC methods — selected by fixed-window "
                f"QDC in [{fixed_qdc_low:g}, {fixed_qdc_high:g}] mV ns "
                f"({fixed_qdc_population})"
            )
            figure.tight_layout(rect=(0, 0, 1, 0.98))
            save(figure)

        def plot_pedestal_examples(records, title, color, *, local_rms=False):
            if not records:
                return
            ncols = 2
            nrows = int(np.ceil(len(records) / ncols))
            figure, axes = plt.subplots(
                nrows, ncols, figsize=(14, 4.2 * nrows), squeeze=False,
                sharex=True,
            )
            if local_rms:
                block_edges = np.arange(
                    sample_time[0],
                    sample_time[-1] + 0.5 * cached["pedestal_local_rms_window_ns"],
                    cached["pedestal_local_rms_window_ns"],
                )
            else:
                block_edges = np.linspace(
                    sample_time[0], sample_time[-1],
                    int(cached["pedestal_time_blocks"]) + 1,
                )
            for ax, record in zip(axes.ravel(), records):
                ax.plot(sample_time, record["waveform_mV"], color=color, linewidth=1.0)
                for edge in block_edges[1:-1]:
                    ax.axvline(edge, color="0.5", linestyle=":", alpha=0.6)
                ax.set_title(
                    f"{record['metric_name']}={record['metric']:.3g}; "
                    f"QDC={record['qdc_mV_ns']:.3g} mV ns", fontsize=9,
                )
                ax.set(
                    xlabel="Time [ns]", ylabel="Median-centered voltage [mV]"
                )
                ax.grid(True, alpha=0.3)
            for ax in axes.ravel()[len(records):]:
                ax.set_visible(False)
            figure.suptitle(title)
            figure.tight_layout(rect=(0, 0.04, 1, 0.96))
            save(figure)

        plot_pedestal_examples(
            pedestal_rejected_records,
            f"Rejected pedestals: noise-growth ratio > {pedestal_growth_ratio_max:g}",
            "tab:orange",
        )
        plot_pedestal_examples(
            pedestal_boundary_records,
            "Highest-growth pedestals still accepted below ratio "
            f"{pedestal_growth_ratio_max:g}",
            "tab:green",
        )
        plot_pedestal_examples(
            pedestal_rms_rejected_records,
            "Rejected pedestals: local RMS relative span > "
            f"{pedestal_rms_variation_max:.1%}",
            "tab:red",
            local_rms=True,
        )
        plot_pedestal_examples(
            pedestal_rms_boundary_records,
            "Highest local-RMS span still accepted below "
            f"{pedestal_rms_variation_max:.1%}",
            "tab:olive",
            local_rms=True,
        )
    temporary_path.replace(output_path)
    print(
        f"Saved {pages}-page waveform-shape diagnostics PDF from reduced cache "
        f"to {output_path}"
    )
    return {
        "output_path": output_path,
        "pages": pages,
        "signal_mismatch_threshold": mismatch_threshold,
        "signal_kept": int(np.sum(signal_keep)),
        "signal_total": len(signal_keep),
        "pedestal_kept": int(np.sum(pedestal_keep)),
        "pedestal_total": len(pedestal_keep),
        "pedestal_rms_constancy_kept": int(np.sum(pedestal_rms_keep)),
        "pedestal_rms_constancy_total": len(pedestal_rms_keep),
    }


def discover_testing_caches(
    fit_data_dir,
    *,
    cache_glob="*_Dark_20microV_trig_df.pkl",
):
    """Return one unselected dataframe cache per acquisition, voltage-sorted."""

    fit_data_dir = Path(fit_data_dir)
    if not fit_data_dir.is_dir():
        raise FileNotFoundError(f"Testing cache directory does not exist: {fit_data_dir}")
    caches = [
        path for path in fit_data_dir.glob(cache_glob)
        if "_selected" not in path.stem
    ]
    if not caches:
        raise FileNotFoundError(
            f"No testing caches matching {cache_glob!r} in {fit_data_dir}"
        )
    caches.sort(key=lambda path: (extract_pmt_voltage(path.name), path.name))
    voltages = [extract_pmt_voltage(path.name) for path in caches]
    if len(set(voltages)) != len(voltages):
        raise ValueError(
            f"Cache pattern {cache_glob!r} resolves duplicate voltages: {voltages}"
        )
    return caches


def _testing_cell_source(notebook_path):
    notebook = nbformat.read(Path(notebook_path), as_version=4)
    cells = [cell for cell in notebook.cells if cell.id == "sample-population-qdc"]
    if len(cells) != 1:
        raise ValueError(
            f"Expected exactly one sample-population-qdc cell in {notebook_path}"
        )
    compile(cells[0].source, f"{notebook_path}:sample-population-qdc", "exec")
    return cells[0].source


@contextmanager
def _capture_shown_figures(pdf):
    original_show = plt.show
    pages_written = 0
    plt.close("all")

    def save_and_close(*args, **kwargs):
        nonlocal pages_written
        for figure_number in tuple(plt.get_fignums()):
            figure = plt.figure(figure_number)
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            pages_written += 1

    plt.show = save_and_close
    try:
        yield lambda: pages_written
        save_and_close()
    finally:
        plt.show = original_show
        plt.close("all")


def generate_testing_summary_pdf(
    cache_path,
    output_path,
    *,
    testing_notebook_path="testing.ipynb",
):
    """Run the testing diagnostics for one cache and save every figure to PDF."""

    cache_path = Path(cache_path).resolve()
    output_path = Path(output_path)
    notebook_path = Path(testing_notebook_path).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Testing cache does not exist: {cache_path}")
    source = _testing_cell_source(notebook_path)
    acquisition_suffix = "_df.pkl"
    if not cache_path.name.endswith(acquisition_suffix):
        raise ValueError(f"Cannot infer acquisition from cache name: {cache_path.name}")
    acquisition = cache_path.name[: -len(acquisition_suffix)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()
    run_globals = {
        "__name__": "__testing_summary__",
        "testing_sample_acquisition_override": acquisition,
        "testing_sample_cache_path_override": cache_path,
    }
    try:
        with PdfPages(temporary_path) as pdf:
            with _capture_shown_figures(pdf) as page_count:
                exec(compile(source, str(notebook_path), "exec"), run_globals)
        pages_written = page_count()
        if pages_written < 1:
            raise RuntimeError(f"Testing produced no figures for {acquisition}")
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        run_globals.clear()
        gc.collect()
    print(f"Saved {pages_written}-page testing summary to {output_path}")
    return {"acquisition": acquisition, "output_path": output_path, "pages": pages_written}


def generate_testing_summaries(
    fit_data_dir,
    output_dir,
    *,
    cache_glob="*_Dark_20microV_trig_df.pkl",
    testing_notebook_path="testing.ipynb",
):
    """Generate one testing-diagnostic PDF for every voltage cache in a directory."""

    caches = discover_testing_caches(fit_data_dir, cache_glob=cache_glob)
    output_dir = Path(output_dir)
    records = []
    print(f"Found {len(caches)} testing caches in {Path(fit_data_dir)}")
    for index, cache_path in enumerate(caches, start=1):
        acquisition = cache_path.name.removesuffix("_df.pkl")
        output_path = output_dir / f"{acquisition}_testing_summary.pdf"
        print("=" * 80)
        print(f"[{index}/{len(caches)}] {acquisition}")
        records.append(
            generate_testing_summary_pdf(
                cache_path,
                output_path,
                testing_notebook_path=testing_notebook_path,
            )
        )
    return records


__all__ = [
    "generate_shape_diagnostics_pdf_from_cache",
    "discover_testing_caches",
    "generate_testing_summary_pdf",
    "generate_testing_summaries",
    "load_shape_diagnostics_cache",
    "save_shape_diagnostics_cache",
]
