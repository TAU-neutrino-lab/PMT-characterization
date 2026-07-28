"""Utilities for combining SPE fits into a PMT high-voltage calibration."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


_RESULT_FILENAME = re.compile(
    r"^(?P<pmt_id>.+?)_(?P<voltage_V>\d+(?:\.\d+)?)V_"
    r"(?P<acquisition>.+?)_"
    r"(?P<integration>full_waveform|led_window)_"
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


__all__ = [
    "load_voltage_scan_results",
    "select_voltage_scan",
    "fit_gain_power_law",
    "evaluate_gain_power_law",
    "plot_gain_calibration",
    "plot_fit_parameters_vs_voltage",
]
