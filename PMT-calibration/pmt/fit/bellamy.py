
import numpy as np
from scipy.special import gammaln

from .core import FitModel
from .pdfs import gaussian_pdf, exponential_gaussian_pdf
from .core import fit_histogram_model


def fit_bellamy_spe(
    charge_mV_ns,
    p0,
    fit_range=None,
    bins=250,
    max_pe=8,
    maxfev=100000
):
    return fit_histogram_model(
        charge_mV_ns,
        model=BELLAMY_SPE,
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        model_kwargs={
            "max_pe": max_pe,
        },
        maxfev=maxfev
    )


def bellamy_spe_model(
    x,
    n_total,
    mu_pe,
    q0_mV_ns,
    sigma0_mV_ns,
    q1_mV_ns,
    sigma1_mV_ns,
    w,
    alpha,
    bin_width,
    max_pe: int = 8,
):
    """Bellamy et al. PMT response model.

    This is the Poisson-weighted PM response where each n-PE Gaussian response is
    mixed with an exponential-background convolution. ``background_probability``
    is the paper's w parameter and ``background_slope`` is the exponential
    coefficient a.
    """
    y = np.zeros_like(x, dtype=float)
    mu_pe = np.maximum(mu_pe, 1e-12)
    w = np.clip(w, 0.0, 1.0)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        qn_mV_ns = q0_mV_ns + n_pe * q1_mV_ns
        sigman_mV_ns = np.sqrt(sigma0_mV_ns**2 + n_pe * sigma1_mV_ns**2)
        response = (1.0 - w) * gaussian_pdf(x, qn_mV_ns, sigman_mV_ns)
        response += w * exponential_gaussian_pdf(x, qn_mV_ns, sigman_mV_ns, alpha)
        y += n_total * bin_width * weight * response

    return y

def bellamy_spe_components(x, parameters, bin_width, max_pe: int = 8):
    """Return Bellamy model components grouped by n photoelectrons."""
    components = {}
    p = parameters
    mu_pe = np.maximum(p["mu_pe"], 1e-12)
    w = np.clip(p["w"], 0.0, 1.0)

    for n_pe in range(max_pe + 1):
        log_weight = -mu_pe + n_pe * np.log(mu_pe) - gammaln(n_pe + 1)
        weight = np.exp(log_weight)
        mean = p["q0_mV_ns"] + n_pe * p["q1_mV_ns"]
        sigma = np.sqrt(p["sigma0_mV_ns"] ** 2 + n_pe * p["sigma1_mV_ns"] ** 2)
        response = (1.0 - w) * gaussian_pdf(x, mean, sigma)
        response += w * exponential_gaussian_pdf(x, mean, sigma, p["alpha"])
        components[n_pe] = p["n_total"] * bin_width * weight * response

    return components

def bellamy_npe_fractions( fit, max_pe=10 ):
    # same as Poisson model
    p = fit["parameters"]
    mu = p["mu_pe"]
    fractions = {}
    for n in range(max_pe + 1):

        fractions[n] = np.exp( -mu + n*np.log(mu) - gammaln(n+1) )

    return fractions

BELLAMY_SPE = FitModel(
    name="bellamy_spe",
    # paper: distinuguish 2 types of noise
    # 1. low charge process present in each event (leakage current, etc.) contributing to the pedestal → Gaussian
    # 2. discrete processes which can, with nonzero probability, accompany the measured signal (thermoemission, noise initiated by the measured light, etc.) → Exponential

    parameter_names=(
        "n_total",
        "mu_pe",
        "q0_mV_ns",
        "sigma0_mV_ns", # std of the type 1 bkg distribution
        "q1_mV_ns",
        "sigma1_mV_ns",
        "w", # probability that a measured signal is accompanied by a type-2 bkg process ()
        "alpha" # coeff. of the exponential decay of type 2-background (positive)
    ),

    model = bellamy_spe_model,

    components = bellamy_spe_components,

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
        "w": r"w",
        "alpha": r"\alpha",
    },
)