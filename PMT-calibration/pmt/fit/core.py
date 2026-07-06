import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import yaml


@dataclass(frozen=True)
class FitModel:
    """
    Description of a fit model.

    Parameters
    ----------
    """

    name: str
    parameter_names: tuple[str, ...]
    model: Callable
    components: Callable | None = None
    default_kwargs: dict = field( default_factory=dict )
    pretty_parameter_names: dict = field(default_factory=dict)
    fractions: Callable | None = None

    def pretty_name(self, parameter_name: str) -> str:
        return self.pretty_parameter_names.get(parameter_name, parameter_name)



def compute_gain(Q1_mV_ns, Q1_mV_ns_err,  oscilloscpe_impedance_ohms=50):
    e = 1.60217663e-19# Coulombs
    QSPE = (Q1_mV_ns*1e-3*1e-9)/oscilloscpe_impedance_ohms# Coulombs
    gain = QSPE/e
    gain_err = ((Q1_mV_ns_err*1e-3*1e-9)/oscilloscpe_impedance_ohms)/e
    return gain, gain_err

# -------------------------------------
#             Fit Functions
# -------------------------------------

def fit_histogram_model( charge_mV_ns, model: FitModel, p0, fit_range=None, bins=250, model_kwargs=None, maxfev=100000):

    charge_mV_ns = np.asarray(charge_mV_ns, dtype=float)
    charge_mV_ns = charge_mV_ns[np.isfinite(charge_mV_ns)]
    if fit_range is None:
        fit_range = (np.min(charge_mV_ns), np.max(charge_mV_ns))
        print(f'using full variable range for fit: ({np.min(charge_mV_ns)}, {np.max(charge_mV_ns)})')
    model_kwargs = {
        **model.default_kwargs,
        **(model_kwargs or {}),
    }

    counts, edges = np.histogram(charge_mV_ns, bins=bins, range=fit_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    parameter_names = ( model.parameter_names )
    specs = parse_fit_parameter_specs( p0, parameter_names )

    initial_parameters     = specs["initial_parameters"]
    fitted_parameter_names = specs["fitted_parameter_names"]

    def full_parameters(fitted_values):
        parameters = dict(initial_parameters)
        parameters.update(zip(fitted_parameter_names, fitted_values))
        return parameters

    def model_from_dict(x, parameters):
        ordered = [parameters[name] for name in parameter_names]
        return model.model(x, *ordered, bin_width=bin_width, **model_kwargs)

    sigma_counts = np.sqrt(np.maximum(counts, 1))

    if fitted_parameter_names:
        def model_for_fit(x, *fitted_values):
            return model_from_dict(x, full_parameters(fitted_values))

        popt_free, pcov_free = curve_fit(
            model_for_fit,
            centers,
            counts,
            p0=specs["fitted_initial"],
            bounds=specs["fitted_bounds"],
            sigma=sigma_counts,
            absolute_sigma=True,
            maxfev=maxfev,
        )
        parameters = full_parameters(popt_free)
    else:
        popt_free = np.array([], dtype=float)
        pcov_free = np.empty((0, 0), dtype=float)
        parameters = dict(initial_parameters)

    model_counts = model_from_dict(centers, parameters)
    chi2 = np.sum(((counts - model_counts) / sigma_counts) ** 2)
    ndof = len(counts) - len(fitted_parameter_names)

    errors = {name: 0.0 if name in specs["fixed_parameters"] else np.nan for name in parameter_names}
    for i, name in enumerate(fitted_parameter_names):
        errors[name] = float(np.sqrt(pcov_free[i, i])) if pcov_free.size else np.nan

    diagnostics = []
    lower = specs["bounds"]["lower"]
    upper = specs["bounds"]["upper"]
    for name in fitted_parameter_names:
        value = parameters[name]
        low = lower[name]
        high = upper[name]
        if np.isclose(value, low, rtol=0, atol=max(1e-6, 1e-3 * max(abs(low), 1.0))):
            diagnostics.append(f"{name} is close to the lower fit bound")
        if np.isfinite(high) and np.isclose(value, high, rtol=0, atol=max(1e-6, 1e-3 * max(abs(high), 1.0))):
            diagnostics.append(f"{name} is close to the upper fit bound")

    return {
        "model": model,
        "model_name": model.name,
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "bin_width": bin_width,
        "fit_range": fit_range,
        "parameter_names": list(parameter_names),
        "fitted_parameter_names": fitted_parameter_names,
        "fixed_parameters": specs["fixed_parameters"],
        "parameters": parameters,
        "errors": errors,
        "covariance": pcov_free,
        "model_counts": model_counts,
        "chi2": float(chi2),
        "ndof": int(ndof),
        "diagnostics": diagnostics,
        "initial_parameters": initial_parameters,
        "bounds": specs["bounds"],
        "model_kwargs": dict(model_kwargs),
    }

def parse_fit_parameter_specs(p0, parameter_names):
    """Parse ``{name: [initial, lower, upper, is_fixed]}`` fit specs.

    Fixed parameters are included in the model parameter dictionary but are not
    varied by ``curve_fit``.
    """
    if p0 is None:
        raise ValueError(
            "p0 must be provided as {param_name: [initial, lower_bound, upper_bound, is_fixed]}"
        )

    missing = [name for name in parameter_names if name not in p0]
    if missing:
        raise KeyError(f"Missing p0 entries for: {', '.join(missing)}")

    unknown = [name for name in p0 if name not in parameter_names]
    if unknown:
        raise KeyError(f"Unknown p0 entries for this model: {', '.join(unknown)}")

    initial_parameters = {}
    bounds_lower = {}
    bounds_upper = {}
    fixed_parameters = {}
    fitted_parameter_names = []
    fitted_initial = []
    fitted_lower = []
    fitted_upper = []

    for name in parameter_names:
        spec = p0[name]
        if isinstance(spec, dict):
            initial = spec["initial"]
            lower = spec["lower"]
            upper = spec["upper"]
            is_fixed = spec.get("is_fixed", spec.get("fixed", False))
        else:
            values = list(spec)
            if len(values) != 4:
                raise ValueError(
                    f"p0[{name!r}] must be [initial, lower_bound, upper_bound, is_fixed]"
                )
            initial, lower, upper, is_fixed = values

        initial = float(initial)
        lower = float(lower)
        upper = float(upper)
        is_fixed = bool(is_fixed)

        if lower > upper:
            raise ValueError(f"Lower bound is above upper bound for {name!r}")
        if is_fixed:
            lower = initial
            upper = initial
        elif not (lower <= initial <= upper):
            raise ValueError(
                f"Initial value for {name!r} is outside its bounds: {initial} not in [{lower}, {upper}]"
            )

        initial_parameters[name] = initial
        bounds_lower[name] = lower
        bounds_upper[name] = upper

        if is_fixed:
            fixed_parameters[name] = initial
        else:
            fitted_parameter_names.append(name)
            fitted_initial.append(initial)
            fitted_lower.append(lower)
            fitted_upper.append(upper)

    return {
        "initial_parameters": initial_parameters,
        "bounds": {
            "lower": bounds_lower,
            "upper": bounds_upper,
        },
        "fixed_parameters": fixed_parameters,
        "fitted_parameter_names": fitted_parameter_names,
        "fitted_initial": np.asarray(fitted_initial, dtype=float),
        "fitted_bounds": (
            np.asarray(fitted_lower, dtype=float),
            np.asarray(fitted_upper, dtype=float),
        ),
    }


# -------------------------------------
#             Helpers
# -------------------------------------

def binning_diagnostics( values, bins, fit_range=None, verbose=True ):
    """
    Inspect histogram binning quality.
    """

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if fit_range is None:
        fit_range = ( np.min(values), np.max(values) )

    counts, edges = np.histogram(
        values,
        bins=bins,
        range=fit_range,
    )

    info = {
        "n_entries": len(values),
        "n_bins": bins,
        "bin_width": edges[1] - edges[0],
        "min_count": int(np.min(counts)),
        "max_count": int(np.max(counts)),
        "mean_count": float(np.mean(counts)),
        "median_count": float(np.median(counts)),
        "n_empty_bins": int(np.sum(counts == 0)),
        "fraction_empty_bins": float( np.mean(counts == 0) ),
        "n_bins_lt_5": int( np.sum(counts < 5) ),
        "fraction_bins_lt_5": float( np.mean(counts < 5) ),
    }

    if verbose:
        print("Binning diagnostics")
        print("-" * 40)
        print( f"Entries              : {info['n_entries']}" )
        print( f"Bins                 : {info['n_bins']}" )
        print( f"Bin width            : {info['bin_width']:.4g}" )
        print( f"Min count/bin        : {info['min_count']}" )
        print( f"Max count/bin        : {info['max_count']}" )
        print( f"Mean count/bin       : {info['mean_count']:.2f}" )
        print( f"Median count/bin     : {info['median_count']:.2f}" )
        print( f"Empty bins           : " f"{info['n_empty_bins']} " f"({100*info['fraction_empty_bins']:.1f}%)" )
        print( f"Bins with <5 counts  : " f"{info['n_bins_lt_5']} " f"({100*info['fraction_bins_lt_5']:.1f}%)" )

    return info

def scan_binning( values, bins_list, fit_range=None  ):

    rows = []
    for bins in bins_list:
        info = binning_diagnostics( values, bins=bins, fit_range=fit_range, verbose=False )
        rows.append(info)

    print( f"{'Bins':>6}" f"{'Width':>10}" f"{'Mean':>10}" f"{'Median':>10}" f"{'Empty%':>10}" f"{'<5%':>10}" f"{'Min':>8}" f"{'Max':>8}" )
    print("-" * 72)
    for row in rows:
        print( f"{row['n_bins']:6d}" f"{row['bin_width']:10.3g}" f"{row['mean_count']:10.1f}" f"{row['median_count']:10.1f}" f"{100*row['fraction_empty_bins']:10.1f}" f"{100*row['fraction_bins_lt_5']:10.1f}" f"{row['min_count']:8d}" f"{row['max_count']:8d}" )

    return rows

def fit_result_table(fit, as_dataframe=True):
    """Summarize initial values, bounds, fixed flags, values, and errors.
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
    model = fit["model"]
    for name in parameter_names:
        rows.append(
            {
                "parameter_latex": model.pretty_name(name),
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
        return pd.DataFrame(rows)
    return rows

# -------------------------------------
#             Display
# -------------------------------------

def print_fit_result_table(fit):

    table = fit_result_table( fit, as_dataframe=False )

    def fmt(x):
        if np.isnan(x):
            return "nan"
        return f"{x:.4f}"

    print( f"{'Parameter':<15}" f"{'Initial':>12}" f"{'Fit':>12}" f"{'Error':>12}" f"{'Bounds':>22}" f"{'Fixed':>8}" )
    print("-" * 81)

    for row in table:
        bounds = ( f"[{row['lower_bound']:.1f}, " f"{row['upper_bound']:.1f}]" )
        print( f"{row['parameter']:<15}" f"{fmt(row['initial']):>12}" f"{fmt(row['fit_value']):>12}" f"{fmt(row['fit_error']):>13}" f"{bounds:>23}" f"{str(row['fixed']):>8}" )

def print_npe_fractions( fractions, n_total=None ):
    """
    Pretty-print PE fractions.

    Parameters
    ----------
    fractions : dict
        {n_pe: fraction}

    n_total : float or None
        If provided, also print the expected number of events.
    """

    print("Photoelectron fractions")
    print("-" * 40)

    for n_pe in sorted(fractions):

        frac = fractions[n_pe]

        if n_total is None:

            print( f"{n_pe:2d} PE : " f"{100*frac:7.3f}%" )

        else:

            print( f"{n_pe:2d} PE : " f"{100*frac:7.3f}%   " f"({n_total*frac:10.1f} events)" )

    print("-" * 40)

    print( f"sum = {100*sum(fractions.values()):.3f}%" )

# -------------------------------------
#             Plots
# -------------------------------------

def plot_correlation_matrix(fig, ax, fit, shrink_colorbar=0.8):

    cov = fit["covariance"]
    names = fit["fitted_parameter_names"]
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    labels = [ rf"${fit['model'].pretty_name(name)}$" for name in names]

    im = ax.imshow( corr, vmin=-1, vmax=1, cmap="coolwarm" )
    ax.set_aspect("equal")
    ax.set_xticks( np.arange(len(names)) )
    ax.set_yticks( np.arange(len(names)) )
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text( j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=8 )

    fig.colorbar( im,ax=ax,label="Correlation coefficient", shrink=shrink_colorbar )
    ax.set_title( fit["model_name"] )

def plot_fit_result(
    fit,
    title="",
    component_function=None,
    component_kwargs=None,
    ax=None,
    ax_resid=None,
    show_components=True,
    show_errorbars=True,
    logscale=True,
    ylims=None,
    residual_ylim=(-8, 8),
    component_visibility_fraction=1e-3,
    show_event_fractions=False,
    max_fraction_pe=100
):
    """Plot histogram data, total fit, model components, and residuals.

    Parameters
    ----------
    fit
        Fit dictionary returned by ``fit_histogram_model`` or one of its model
        wrappers.
    component_function
        Optional callable with signature ``f(x, parameters, bin_width, **kwargs)``.
        If omitted, the function is chosen automatically for known models.
    component_kwargs
        Extra keyword arguments passed to ``component_function``. By default,
        ``max_pe`` is taken from the fit dictionary when available.
    """

    if ax is None or ax_resid is None:
        fig, (ax, ax_resid) = plt.subplots( 2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    else:
        fig = ax.figure

    centers = fit["centers"]
    counts = fit["counts"]
    model_counts = fit["model_counts"]
    counts_unc = np.sqrt(np.maximum(counts, 1))
    residuals = (counts - model_counts) / counts_unc

    if show_errorbars:
        ax.errorbar( centers, counts, yerr=counts_unc,
                     fmt=".", ms=3, color="black", alpha=0.75,
                     label="data",
        )
    else:
        ax.plot(centers, counts, ".", ms=3, color="black", alpha=0.75, label="data")

    ax.plot(centers, model_counts, color="tab:red", lw=2.2, label="total fit")

    # --------- Components ---------
    if show_components:
        component_function = ( component_function or fit["model"].components )

        if component_function is not None:
            component_kwargs = { **fit.get("model_kwargs", {}), **(component_kwargs or {}) }
            components = component_function( centers, fit["parameters"], fit["bin_width"], **component_kwargs )
            try:
                component_items = sorted( components.items(), key=lambda kv: kv[0] )
            except Exception:
                component_items = list( components.items() )

            # Skip tiny components (optional)
            total_area = np.sum( model_counts )
            component_visibility_threshold = ( component_visibility_fraction * total_area )
            for key, y_component in component_items:
                component_area = np.sum( y_component )
                if ( component_area < component_visibility_threshold ):
                    continue
                if isinstance( key, (int, np.integer) ):
                    label = f"{key} PE"
                else:
                    label = str(key)

                y_plot = np.asarray( y_component, dtype=float ).copy()
                if logscale: 
                    threshold = 1e-6 * np.nanmax(y_plot)
                    y_plot[y_plot < threshold] = np.nan
                
                ax.plot( centers, y_plot, ls="--", lw=1.4, alpha=0.8, label=label )

                # ax.plot( centers, y_component, ls="--", lw=1.4, alpha=0.8, label=label )

    ax.set_title(title)
    ax.set_ylabel("Events / bin")
    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    if ylims is not None:
        ax.set_ylim(ylims)

    # --------- Fit summary box ---------
    parameters = fit["parameters"]
    errors = fit["errors"]
    chi2_ndof = ( fit["chi2"] / fit["ndof"] if fit["ndof"] > 0 else np.nan )
    summary_lines = [ rf"$\chi^2/\mathrm{{ndof}} = {chi2_ndof:.2f}$" ]
    model = fit["model"]

    for par in fit["parameter_names"]:
        if par not in parameters:
            continue

        pretty_name = model.pretty_name(par)
        value = parameters[par]
        error = errors[par]

        if par in fit["fixed_parameters"]:
            summary_lines.append( rf"${pretty_name} = {value:.3f}$ (fixed)" )
        else:
            summary_lines.append( rf"${pretty_name} = " rf"{value:.3f}" rf" \pm {error:.2f}$" )

    summary_text = "\n".join(summary_lines)

    ax.text( 0.98, 0.98, summary_text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict( boxstyle="round", facecolor="white", alpha=0.85 ) )
    
        # --------- Event fractions ---------

    if ( show_event_fractions and fit["model"].fractions is not None ):
        fractions = fit["model"].fractions( fit, max_pe=max_fraction_pe )
        fraction_lines = [ "PE fractions" ]

        for n_pe, frac in fractions.items():
            fraction_lines.append( f"{n_pe} PE : {100*frac:.2f}%" )

        fraction_text = "\n".join( fraction_lines )

        ax.text(
            0.02,
            0.98,
            fraction_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
            ),
        )


    ax.legend(ncol=2, fontsize=8)
    ax_resid.axhline(0, color="black", lw=1)
    ax_resid.axhline(2, color="0.55", lw=0.8, ls=":")
    ax_resid.axhline(-2, color="0.55", lw=0.8, ls=":")
    if show_errorbars:
        ax_resid.errorbar( centers, residuals, yerr=np.ones_like(residuals),
                           fmt=".", color="tab:gray", ms=3, elinewidth=0.8, capsize=0 )
    else:
        ax_resid.plot(centers, residuals, ".", color="tab:gray", ms=3)
    ax_resid.set_xlabel("Charge [mV ns]")
    ax_resid.set_ylabel("Residual\n[data-fit]/sigma")
    if residual_ylim is not None:
        ax_resid.set_ylim(residual_ylim)

    ax.tick_params( axis="x", which="both", labelbottom=False )

    return fig, ax, ax_resid

def plot_fit_summary( fit, title="", figsize=(20, 8), shrink_colorbar=0.5, **plot_fit_kwargs):
    """
    Plot fit result + residuals + correlation matrix.
    """
    fig      = plt.figure( figsize=figsize )
    gs       = fig.add_gridspec( 2, 2, width_ratios=[3, 2], height_ratios=[3, 1], hspace=0.05, wspace=0.25 )
    ax_fit   = fig.add_subplot( gs[0, 0] )
    ax_resid = fig.add_subplot( gs[1, 0], sharex=ax_fit, )
    ax_corr  = fig.add_subplot( gs[:, 1] )

    plot_fit_result( fit, title=title, ax=ax_fit, ax_resid=ax_resid, **plot_fit_kwargs )
    plot_correlation_matrix( fig, ax_corr, fit, shrink_colorbar=shrink_colorbar  )

    return ( fig, ax_fit, ax_resid, ax_corr )


# -------------------------------------
#          save fit results
# -------------------------------------
def pack_fit_results(fit):
    return {
        name: {
            "value": float(value),
            "error": fit["errors"].get(name),
        }
        for name, value in fit["parameters"].items()
    }

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_python(v) for v in obj]

    elif isinstance(obj, tuple):
        return [to_python(v) for v in obj]   # YAML-friendly list

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.generic):
        return obj.item()

    else:
        return obj
    
def save_fit_results(fit_res, output_path):
    results = {
        "fit_model": fit_res['model_name'],
        "bin_width": fit_res['bin_width'],
        "fit_range": fit_res['fit_range'],
        "parameter_names": fit_res['parameter_names'],
        "fixed_parameters": fit_res['fixed_parameters'],
        "covariance": fit_res['covariance'],
        "chi2": fit_res['chi2'],
        "ndof": fit_res['ndof'],
        "model_params": fit_res['model_kwargs'],
        "fit_results": pack_fit_results(fit_res),
    }
    G, G_err = compute_gain(fit_res['parameters']['q1_mV_ns'], fit_res['errors']['q1_mV_ns'])
    results["gain"] = {
        "value": float(G),
        "error": G_err,
    }
    results = to_python(results)
    with open(output_path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)