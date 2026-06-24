
import numpy as np
from scipy.special import gammaln

from .core import FitModel
from .pdfs import gaussian_pdf
from .core import fit_histogram_model

def fit_poisson_spe(
    charge_mV_ns,
    p0,
    fit_range=None,
    bins=250,
    max_pe=8,
    maxfev=100000
):
    return fit_histogram_model(
        charge_mV_ns,
        model=POISSON_SPE,
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_kwargs={
            "max_pe": max_pe,
        },
        maxfev=maxfev
    )

def poisson_spe_model(
    x,
    n_total,
    mu_pe,
    q0,
    sigma0,
    q1,
    sigma1,
    bin_width,
    max_pe: int = 8,
):
    """Bellamy-style ideal Gaussian PM response.

    Parameter names follow Bellamy notation:
    Q0 -> ``q0_mV_ns``, sigma0 -> ``sigma0_mV_ns``,
    Q1 -> ``q1_mV_ns``, sigma1 -> ``sigma1_mV_ns``.
    """
    y = np.zeros_like(x, dtype=float)
    mu_pe = np.maximum(mu_pe, 1e-12)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = q0 + n_pe * q1
        sigma = np.sqrt(sigma0**2 + n_pe * sigma1**2)
        y += n_total * bin_width * weight * gaussian_pdf(x, mean, sigma)
    return y

def poisson_spe_components( x, parameters, bin_width, max_pe: int = 8 ):
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

def poisson_npe_fractions( fit, max_pe=10 ):
    p = fit["parameters"]
    mu = p["mu_pe"]
    fractions = {}
    for n in range(max_pe + 1):

        fractions[n] = np.exp( -mu + n*np.log(mu) - gammaln(n+1) )

    return fractions

POISSON_SPE = FitModel(
    name="poisson_spe",

    parameter_names=(
        "n_total",
        "mu_pe",
        "q0_mV_ns",
        "sigma0_mV_ns",
        "q1_mV_ns",
        "sigma1_mV_ns",
    ),

    model = poisson_spe_model,

    components = poisson_spe_components,

    default_kwargs={
        "max_pe": 8,
    },
    
    pretty_parameter_names={
        "n_total": r"N_{\mathrm{tot}}",
        "mu_pe": r"\mu_{\mathrm{PE}}",
        "q0_mV_ns": r"Q_0",
        "sigma0_mV_ns": r"\sigma_0",
        "q1_mV_ns": r"Q_1",
        "sigma1_mV_ns": r"\sigma_1",
    },
)




