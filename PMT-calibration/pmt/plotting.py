
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------
# generic plots
# ----------------------------------------------------------------

def add_hist_stats(ax, values, bins, color, xlabel, title, unit="", show_stat=True, text_position=(0.02, 0.98), legend_loc="best"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, histtype="step", lw=1.8, color=color)
    if len(values) and show_stat:
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        q05, q95 = np.quantile(values, [0.05, 0.95])
        label_unit = f" {unit}" if unit else ""
        ax.axvline(mean, color="tab:orange", ls=":", lw=1.8, label=f"mean = {mean:.4g}{label_unit}")
        ax.axvline(median, color="black", ls="--", lw=1.3, label=f"median = {median:.4g}{label_unit}")
        ax.axvspan(q05, q95, color=color, alpha=0.30, label=f"5-95% = [{q05:.3g}, {q95:.3g}]{label_unit}")
        ax.text(
            text_position[0],
            text_position[1],
            f"N = {len(values):,}\nstd = {std:.4g}{label_unit}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.set_title(title)
    ax.legend(fontsize=8, loc=legend_loc)

# def add_plot()

# ----------------------------------------------------------------
# dedicated plots
# ----------------------------------------------------------------

def plot_baseline_stability(ax, values, unit="mV", max_points=20000):
    # baseline value in each event
    if max_points is not None:
        step = max(1, len(values) // max_points)
        event_index = np.arange(len(values))[::step]
        ax.plot(event_index, values[::step], ".", ms=2, alpha=0.35, color="tab:green")
    else:
        event_index = np.arange(len(values))
        ax.plot(event_index, values, ".", ms=2, alpha=0.35, color="tab:green")
    ax.axhline(np.mean(values), color="tab:orange", ls=":", lw=1.5, label="mean")
    ax.axhline(np.median(values), color="black", ls="--", lw=1.2, label="median")
    ax.set_xlabel("Event index")
    ax.set_ylabel(f"Baseline [{unit}]")
    title = "Baseline stability vs event index"
    if max_points is not None:
        title += f"every {step} event(s))"
    ax.set_title(title)
    ax.legend(fontsize=10)

# ----------------------------------------------------------------
# plots wrappers
# ----------------------------------------------------------------

def plot_baseline_quality(axes, time_ns, sample_baseline_mV, voltage_sample_bs_mV, baseline_window_ns):
    axes = axes.ravel()
    add_hist_stats(
        axes[0],
        sample_baseline_mV,
        bins=80,
        color="tab:green",
        xlabel="Baseline estimate [mV]",
        title="Baseline distribution before subtraction",
        unit="mV",
        show_stat=True,
        text_position=(0.02, 0.98),
        legend_loc="lower center"
    )

    check_mask = (time_ns >= baseline_window_ns[0]) & (time_ns <= baseline_window_ns[1])
    baseline_means = [np.mean(voltage_sample_bs_mV[:, check_mask], axis=1)]
    add_hist_stats(
            axes[1],
            baseline_means,
            bins=120,
            color="tab:blue",
            xlabel="Mean voltage in baseline window [mV]",
            title="Baseline removal residual",
            unit="mV",
        )

    plot_baseline_stability(axes[2], sample_baseline_mV, unit="mV", max_points=None)


# ----------------------------------------------------------------
# save
# ----------------------------------------------------------------
def save_plot(fig, save_plots=False, save_dir="plots", file_nickname="plot", plot_name="figure", formats=("pdf",)):
    """Save a figure when requested and return saved paths."""
    if not save_plots:
        return []
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = save_path / f"{file_nickname}_{plot_name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    return saved