import numpy as np

from .core import FitModel
from .pdfs import exponential_gaussian_pdf
from .core import fit_histogram_model


def fit_pedestal_expgauss( charge_mV_ns, p0, fit_range=None, bins=250, maxfev=100000 ):
   
    return fit_histogram_model(
        charge_mV_ns,
        model=PEDESTAL_EXPGAUSS,
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        maxfev=maxfev,
    )


def expgauss_pedestal_model( x, n_total, q0_mV_ns, sigma0_mV_ns, alpha, bin_width ):
     # Pedestal = ExGaussian(Q0, σ0, α)
    response = ( exponential_gaussian_pdf( x, q0_mV_ns, sigma0_mV_ns, alpha  ) )

    return n_total * bin_width * response


def expgauss_pedestal_components( x, parameters, bin_width,):
    p = parameters
    expgauss_part = ( exponential_gaussian_pdf( x, p["q0_mV_ns"], p["sigma0_mV_ns"], p["alpha"]  ) )

    return { 
        "expgaussian": ( p["n_total"] * bin_width * expgauss_part ),
     }


PEDESTAL_EXPGAUSS = FitModel(
    name="pedestal exponential Gaussian",

    parameter_names=(
        "n_total",
        "q0_mV_ns",
        "sigma0_mV_ns",
        'alpha'
    ),

    model=expgauss_pedestal_model,

    components=expgauss_pedestal_components,

    pretty_parameter_names={
        "n_total": r"N_{\mathrm{tot}}",
        "q0_mV_ns": r"Q_0",
        "sigma0_mV_ns": r"\sigma_0",
        "alpha": r"\alpha",
    },
)