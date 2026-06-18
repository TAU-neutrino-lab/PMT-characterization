import matplotlib.pyplot as plt
import numpy as np



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