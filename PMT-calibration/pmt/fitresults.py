
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import yaml

from scipy.optimize import curve_fit

# ----------------------------------------------------------------
# read yaml
# ----------------------------------------------------------------

def parse_filename(path):
    name = Path(path).stem

    m = re.match(
        r'^(?P<pmt>[^_]+)_(?P<HV>\d+)V_(?P<BW>\d+)MHz_led(?P<LED>\d+)_(?P<model>.+)$',
        name
    )

    if m is None:
        raise ValueError(f"Cannot parse filename: {name}")

    d = m.groupdict()
    d["HV"] = int(d["HV"])
    d["BW"] = int(d["BW"])
    d["LED"] = int(d["LED"])

    return d


def load_fit_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f)

    row = parse_filename(path)

    row.update({
        "chi2": d["chi2"],
        "ndof": d["ndof"],
        "chi2_ndof": d["chi2"] / d["ndof"],
        "gain": d["gain"]["value"],
        "gain_err": d["gain"]["error"],
        "file": Path(path).stem
    })

    for par, vals in d["fit_results"].items():
        row[par] = vals["value"]
        row[f"{par}_err"] = vals["error"]
        if vals.get("initial") is not None:
            row[f"{par}_initial"] = vals["initial"]
        if vals.get("lower_bound") is not None:
            row[f"{par}_lower"] = vals["lower_bound"]
        if vals.get("upper_bound") is not None:
            row[f"{par}_upper"] = vals["upper_bound"]

    return row

# ----------------------------------------------------------------
# plot
# ----------------------------------------------------------------

def plot_parameter( df, parameter, ax=None, x="HV", xlabel=None, ylabel=None, title=None, label=None, show_errors=True, show_bounds=False, bound_alpha=0.15, **errorbar_kwargs):

    df = df.sort_values(x)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    defaults = dict(fmt="o-", capsize=3)
    defaults.update(errorbar_kwargs)

    if show_errors:
        yerr = df.get(f"{parameter}_err")
    else:
        yerr = None

    ax.errorbar( df[x], df[parameter], yerr=yerr, label=label, **defaults)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or parameter)

    if show_bounds:
        lower = df.get(f"{parameter}_lower")
        upper = df.get(f"{parameter}_upper")
        if lower is not None and upper is not None:
            # Same color as the plotted line
            color = ax.lines[-1].get_color()
            ax.fill_between( df[x], lower, upper, color=color, alpha=bound_alpha, linewidth=3, linestyle='--', zorder=0 )

    if title is not None:
        ax.set_title(title)

    if label is not None:
        ax.legend()
    return ax


def plot_fit_res( ax, ax_res, df, fit, title=r"$G(V)=G_{1000}(V/1000)^k$", label="Power-law fit", plot_log=False, 
                 residual_type="absolute",   # "absolute", "relative", or "pull"
                 ):

    # ---------- Main plot ----------
    plot_parameter( df, "gain", ax, ylabel="Gain", xlabel="HV [V]", ms=2 )
    plot_gain_fit( fit, df, ax=ax, color="red", lw=2, label=label )

    ax.set_title(title)

    summary_text = (
        f"G1000 = {fit['parameters']['G1000']:.3g}\n"
        f"k = {fit['parameters']['k']:.2f}\n"
        f"$\\chi^2$/ndof = {fit['chi2_ndof']:.2f}"
    )

    ax.text( 0.25, 0.80, summary_text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend()

    # ---------- Residuals ----------
    gain_fit = gain_model( df["HV"], fit["parameters"]["G1000"], fit["parameters"]["k"])

    residuals = df["gain"] - gain_fit

    if residual_type == "absolute":
        y = residuals
        yerr = df.get("gain_err")
        ylabel = "Residual"

    elif residual_type == "relative":
        y = residuals / df["gain"]
        yerr = ( df["gain_err"] / df["gain"] if "gain_err" in df else None)
        ylabel = "Relative residual"

    elif residual_type == "pull":
        if "gain_err" not in df:
            raise ValueError("gain_err column required for pull plot.")

        y = residuals / df["gain_err"]
        yerr = np.ones_like(y)
        ylabel = "Pull"

    else:
        raise ValueError( "residual_type must be " "'absolute', 'relative' or 'pull'.")

    ax_res.errorbar( df["HV"], y, yerr=yerr, fmt="o", ms=2, capsize=3 )

    ax_res.axhline(0, color="k", ls="--")

    ax_res.set_ylabel(ylabel)
    ax_res.set_xlabel("HV [V]")

    if plot_log:
        ax_res.set_xscale("log")
        ax.set_xscale("log")
        ax.set_yscale("log")

def make_fit_figure(figsize=(14, 8), height_ratios=[3, 1], hspace=0.05, wspace=0.25):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec( 2, 2, height_ratios=height_ratios, hspace=hspace, wspace=wspace )
    ax =     [ fig.add_subplot(gs[0, 0]),  fig.add_subplot(gs[0, 1]) ]
    ax_res = [ fig.add_subplot(gs[1, 0], sharex=ax[0]), fig.add_subplot(gs[1, 1], sharex=ax[1]) ]
    return fig, ax, ax_res

# ----------------------------------------------------------------
# fit
# ----------------------------------------------------------------



def gain_model(V, G1000, k):
    return G1000 * (V / 1000.0)**k




def fit_gain_voltage(df, method="curve_fit"):
    """
    Fit the PMT gain-voltage relation
        G(HV) = G1000 * (HV / 1000)^k
    or Log
        log(G) = log(G1000) + k log(HV/1000)
    Parameters
    ----------
    method : {"curve_fit", "log_polyfit"}
    """

    df = df.sort_values("HV")

    if method == "curve_fit":

        popt, pcov = curve_fit(
            gain_model,
            df["HV"],
            df["gain"],
            sigma=df["gain_err"],
            absolute_sigma=True,
            p0=[1e7, 8],
        )

        G1000, k = popt
        G1000_err, k_err = np.sqrt(np.diag(pcov))

    elif method == "log_polyfit":

        x = np.log(df["HV"] / 1000)
        y = np.log(df["gain"])
        coef, pcov = np.polyfit(x, y, 1, cov=True)
        k = coef[0]
        G1000 = np.exp(coef[1])
        k_err = np.sqrt(pcov[0, 0])
        logG1000_err = np.sqrt(pcov[1, 1])
        G1000_err = G1000 * logG1000_err

        popt = np.array([G1000, k])

    else:
        raise ValueError( "method must be 'curve_fit' or 'log_polyfit'" )
    
    # ---------- goodness of fit ----------
    gain_fit = gain_model(df["HV"], G1000, k)
    residuals = df["gain"] - gain_fit
    if "gain_err" in df:
        chi2 = np.sum((residuals / df["gain_err"]) ** 2)
    else:
        chi2 = np.sum(residuals**2)
    ndof = len(df) - 2
    chi2_ndof = chi2 / ndof

    return {
        "method": method,
        "parameters": {
            "G1000": G1000,
            "k": k,
        },
        "errors": {
            "G1000": G1000_err,
            "k": k_err,
        },
        "covariance": pcov,
        "popt": popt,
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof
    }


def plot_gain_fit(fit, df, ax=None, **kwargs):

    df = df.sort_values("HV").reset_index(drop=True)

    if ax is None:
        _, ax = plt.subplots()

    HV = np.linspace(df["HV"].min(), df["HV"].max(), 300)
    G1000 = fit["parameters"]["G1000"]
    k = fit["parameters"]["k"]
    ax.plot( HV, gain_model(HV, G1000, k), **kwargs )

    return ax




