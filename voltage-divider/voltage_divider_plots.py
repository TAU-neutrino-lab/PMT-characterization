"""Plotting helpers for voltage-divider measurement/simulation comparisons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from voltage_divider_data import CUMULATIVE_PIN_ORDER


def _ordered_pin_dataframe(comparison_df, input_voltage, plot_pin_order):
    return (
        comparison_df[comparison_df["input_voltage_V"] == input_voltage]
        .set_index("pins")
        .reindex(plot_pin_order)
        .reset_index()
    )


def _hide_unused_axes(axes, used_count):
    for ax in axes[used_count:]:
        ax.axis("off")


def _pin_voltage_dataframe(data, source, plot_pin_order):
    source = source.lower()

    if source in {"measured", "measurement"}:
        return (
            data.measurement_df.rename(
                columns={
                    "computed_between": "pins",
                    "cumulative_voltage_from_K_V": "pin_voltage_V",
                }
            )[["input_voltage_V", "pins", "pin_voltage_V"]]
        )

    if source in {"simulated", "simulation", "sim"}:
        return (
            data.sim_df.drop(columns="pins")
            .reindex(plot_pin_order)
            .rename_axis(index="pins", columns="input_voltage_V")
            .stack()
            .rename("pin_voltage_V")
            .reset_index()
        )

    raise ValueError('source must be "measured" or "simulated"')


def plot_pin_voltage_vs_input_voltage(
    data,
    source="measured",
    plot_pin_order=None,
    *,
    figsize=(10, 6),
    legend_ncol=3,
    title=None,
    ax=None,
    show=True,
):
    """Plot pin voltage vs input voltage, one line per pin."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    voltage_df = _pin_voltage_dataframe(data, source, plot_pin_order)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for pin in plot_pin_order:
        pin_df = voltage_df[voltage_df["pins"] == pin]
        ax.plot(
            pin_df["input_voltage_V"],
            pin_df["pin_voltage_V"],
            "o-",
            label=pin,
        )

    source_label = "Measured" if source.lower() in {"measured", "measurement"} else "Simulated"
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("Voltage from K [V]")
    ax.set_title(title or f"{source_label} pin voltage vs input voltage")
    ax.legend(ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_pin_voltage_vs_pin(
    data,
    source="measured",
    plot_pin_order=None,
    *,
    figsize=(12, 6),
    legend_ncol=3,
    title=None,
    ax=None,
    show=True,
):
    """Plot pin voltage vs pin, one line per input voltage."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    voltage_df = _pin_voltage_dataframe(data, source, plot_pin_order)
    input_voltages = sorted(voltage_df["input_voltage_V"].unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for input_voltage in input_voltages:
        input_df = (
            voltage_df[voltage_df["input_voltage_V"] == input_voltage]
            .set_index("pins")
            .reindex(plot_pin_order)
            .reset_index()
        )
        ax.plot(
            input_df["pins"],
            input_df["pin_voltage_V"],
            "o-",
            label=f"{input_voltage:.0f} V",
        )

    source_label = "Measured" if source.lower() in {"measured", "measurement"} else "Simulated"
    ax.set_xlabel("Pins")
    ax.set_ylabel("Voltage from K [V]")
    ax.set_title(title or f"{source_label} pin voltage vs pin")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Input voltage", ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_by_pin(
    comparison_df,
    plot_pin_order=None,
    *,
    ncols=4,
    figsize=None,
    show=True,
):
    """Plot measured and simulated voltage vs input voltage, one subplot per pin."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    nrows = int(np.ceil(len(plot_pin_order) / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3.3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, pin in zip(axes, plot_pin_order):
        pin_df = comparison_df[comparison_df["pins"] == pin]

        ax.plot(
            pin_df["input_voltage_V"],
            pin_df["measured_voltage_V"],
            "o-",
            label="Measured",
        )
        ax.plot(
            pin_df["input_voltage_V"],
            pin_df["simulated_voltage_V"],
            "s--",
            label="Simulated",
        )

        ax.set_title(pin)
        ax.set_xlabel("Input voltage [V]")
        ax.set_ylabel("Voltage from K [V]")
        ax.grid(True, alpha=0.3)

    _hide_unused_axes(axes, len(plot_pin_order))
    fig.legend(["Measured", "Simulated"], loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    return fig, axes


def plot_by_input_voltage(
    comparison_df,
    plot_pin_order=None,
    *,
    ncols=4,
    figsize=None,
    show=True,
):
    """Plot measured and simulated voltage vs pin, one subplot per input voltage."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    input_voltages = sorted(comparison_df["input_voltage_V"].unique())
    nrows = int(np.ceil(len(input_voltages) / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 3.3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, input_voltage in zip(axes, input_voltages):
        voltage_df = _ordered_pin_dataframe(comparison_df, input_voltage, plot_pin_order)

        ax.plot(
            voltage_df["pins"],
            voltage_df["measured_voltage_V"],
            "o-",
            label="Measured",
        )
        ax.plot(
            voltage_df["pins"],
            voltage_df["simulated_voltage_V"],
            "s--",
            label="Simulated",
        )

        ax.set_title(f"{input_voltage:.0f} V input")
        ax.set_xlabel("Pins")
        ax.set_ylabel("Voltage from K [V]")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    _hide_unused_axes(axes, len(input_voltages))
    fig.legend(["Measured", "Simulated"], loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    return fig, axes


def plot_residuals_by_voltage(
    comparison_df,
    plot_pin_order=None,
    *,
    residual_column="delta_V",
    figsize=(10, 6),
    legend_ncol=3,
    show=True,
):
    """Plot residuals vs input voltage, one line per pin."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    fig, ax = plt.subplots(figsize=figsize)

    for pin in plot_pin_order:
        pin_df = comparison_df[comparison_df["pins"] == pin]
        ax.plot(
            pin_df["input_voltage_V"],
            pin_df[residual_column],
            "o-",
            label=pin,
        )

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("Measured - simulated [V]")
    ax.set_title("Residuals by input voltage")
    ax.legend(ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_residuals_by_pin(
    comparison_df,
    plot_pin_order=None,
    *,
    residual_column="delta_V",
    figsize=(12, 6),
    legend_ncol=3,
    show=True,
):
    """Plot residuals vs pin, one line per input voltage."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    input_voltages = sorted(comparison_df["input_voltage_V"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    for input_voltage in input_voltages:
        voltage_df = _ordered_pin_dataframe(comparison_df, input_voltage, plot_pin_order)
        ax.plot(
            voltage_df["pins"],
            voltage_df[residual_column],
            "o-",
            label=f"{input_voltage:.0f} V",
        )

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Pins")
    ax.set_ylabel("Measured - simulated [V]")
    ax.set_title("Residuals by pin")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Input voltage", ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_relative_error_by_voltage(
    comparison_df,
    plot_pin_order=None,
    *,
    error_column="signed_relative_error_percent",
    figsize=(10, 6),
    legend_ncol=3,
    show=True,
):
    """Plot signed relative error vs input voltage, one line per pin."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    fig, ax = plt.subplots(figsize=figsize)

    for pin in plot_pin_order:
        pin_df = comparison_df[comparison_df["pins"] == pin]
        ax.plot(
            pin_df["input_voltage_V"],
            pin_df[error_column],
            "o-",
            label=pin,
        )

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Input voltage [V]")
    ax.set_ylabel("Signed relative error [%]")
    ax.set_title("Signed relative error by input voltage")
    ax.legend(ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_relative_error_by_pin(
    comparison_df,
    plot_pin_order=None,
    *,
    error_column="signed_relative_error_percent",
    figsize=(12, 6),
    legend_ncol=3,
    show=True,
):
    """Plot signed relative error vs pin, one line per input voltage."""

    if plot_pin_order is None:
        plot_pin_order = CUMULATIVE_PIN_ORDER

    input_voltages = sorted(comparison_df["input_voltage_V"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    for input_voltage in input_voltages:
        voltage_df = _ordered_pin_dataframe(comparison_df, input_voltage, plot_pin_order)
        ax.plot(
            voltage_df["pins"],
            voltage_df[error_column],
            "o-",
            label=f"{input_voltage:.0f} V",
        )

    ax.axhline(0, color="k", linewidth=1)
    ax.set_xlabel("Pins")
    ax.set_ylabel("Signed relative error [%]")
    ax.set_title("Signed relative error by pin")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Input voltage", ncol=legend_ncol)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
