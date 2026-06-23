"""Reusable PMT LED calibration helpers.

The charge convention used here is:
    charge_mV_ns = - integral(baseline-subtracted voltage_mV dt_ns)

This makes negative-going PMT pulses have positive charge, with units mV ns.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import curve_fit
from scipy.special import erf, erfc, gammaln

from lab_tools.io import iter_keysight_chunks, read_keysight_h5_direct, standard_units


# def poisson_spe_model(
#     x,
#     n_total,
#     mu_pe,
#     q0,
#     sigma0,
#     q1,
#     sigma1,
#     bin_width,
#     max_pe: int = 8,
# ):
#     """Bellamy-style ideal Gaussian PM response.

#     Parameter names follow Bellamy notation:
#     Q0 -> ``q0_mV_ns``, sigma0 -> ``sigma0_mV_ns``,
#     Q1 -> ``q1_mV_ns``, sigma1 -> ``sigma1_mV_ns``.
#     """
#     y = np.zeros_like(x, dtype=float)
#     mu_pe = np.maximum(mu_pe, 1e-12)

#     for n_pe in range(max_pe + 1):
#         log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
#         weight = np.exp(log_weight)
#         mean = q0 + n_pe * q1
#         sigma = np.sqrt(sigma0**2 + n_pe * sigma1**2)
#         y += n_total * bin_width * weight * gaussian_pdf(x, mean, sigma)
#     return y


def poisson_spe_components(
    x,
    parameters,
    bin_width,
    max_pe: int = 8,
):
    """Return the individual n-photoelectron Gaussian components."""
    components = {}
    p = parameters
    mu_pe = np.maximum(p["mu_pe"], 1e-12)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = p["q0_mV_ns"] + n_pe * p["q1_mV_ns"]
        sigma = np.sqrt(p["sigma0_mV_ns"] ** 2 + n_pe * p["sigma1_mV_ns"] ** 2)
        components[n_pe] = p["n_total"] * bin_width * weight * gaussian_pdf(x, mean, sigma)

    return components


def fit_spe_spectrum(
    charge_mV_ns,
    fit_range=None,
    bins: int = 250,
    max_pe: int = 8,
    p0=None,
    maxfev: int = 100000,
):
    """Fit the charge spectrum with the Poisson-weighted SPE model.

    ``p0`` must be ``{param_name: [initial, lower, upper, is_fixed]}``.
    """
    fit = fit_histogram_model(
        charge_mV_ns,
        poisson_spe_model,
        poisson_spe_parameter_names(),
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_name="poisson_spe",
        model_kwargs={"max_pe": max_pe},
        maxfev=maxfev,
    )
    fit["max_pe"] = max_pe
    return fit


def fit_result_table(fit, as_dataframe=True):
    """Summarize initial values, bounds, fixed flags, values, and errors.

    Parameters
    ----------
    fit
        Fit dictionary returned by ``fit_histogram_model`` or one of its model
        wrappers.
    as_dataframe
        If True, return a pandas DataFrame when pandas is available. If pandas
        is not installed, a list of dictionaries is returned instead.
    """
    parameter_names = fit.get("parameter_names")
    if parameter_names is None:
        parameter_names = list(fit.get("parameters", {}).keys())

    initial = fit.get("initial_parameters", {})
    bounds = fit.get("bounds", {})
    lower = bounds.get("lower", {})
    upper = bounds.get("upper", {})
    fixed_parameters = fit.get("fixed_parameters", {})
    parameters = fit.get("parameters", {})
    errors = fit.get("errors", {})

    rows = []
    for name in parameter_names:
        rows.append(
            {
                "parameter": name,
                "initial": initial.get(name, np.nan),
                "lower_bound": lower.get(name, np.nan),
                "upper_bound": upper.get(name, np.nan),
                "fixed": name in fixed_parameters,
                "fit_value": parameters.get(name, np.nan),
                "fit_error": errors.get(name, np.nan),
            }
        )

    if as_dataframe:
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            pass
    return rows



    
# def fit_component_function(fit):
#     """Return the component helper corresponding to a standard fit model."""
#     model = fit.get("model")
#     if model == "poisson_spe":
#         return poisson_spe_components
#     if model == "bellamy_spe":
#         return bellamy_spe_components
#     if model == "dynode_spe":
#         return dynode_spe_components
#     if model == "free_spe":
#         return free_spe_components
#     if model == "backscatter_spe":
#         return backscatter_spe_components
#     return None