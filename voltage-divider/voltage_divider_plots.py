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
