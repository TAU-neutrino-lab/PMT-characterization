import numpy as np

from .core import FitModel
from .pdfs import gaussian_pdf, exponential_gaussian_pdf
from .core import fit_histogram_model


def fit_pedestal_bellamy( charge_mV_ns, p0, fit_range=None, bins=250, maxfev=100000 ):
   
    return fit_histogram_model(
        charge_mV_ns,
        model=PEDESTAL_BELLAMY,
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        maxfev=maxfev,
    )


def bellamy_pedestal_model( x, n_total, q0_mV_ns, sigma0_mV_ns, w, alpha, bin_width ):
     # Pedestal = Gaussian(Q0, σ0) + w ExGaussian(Q0, σ0, α)
    response = ( (1.0 - w) * gaussian_pdf( x, q0_mV_ns, sigma0_mV_ns ) )
    response += (w * exponential_gaussian_pdf( x, q0_mV_ns, sigma0_mV_ns, alpha  ) )

    return n_total * bin_width * response


def bellamy_pedestal_components( x, parameters, bin_width,):
    p = parameters
    gaussian_part = ( (1.0 - p["w"]) * gaussian_pdf( x, p["q0_mV_ns"], p["sigma0_mV_ns"] ) )
    expgauss_part = ( p["w"] * exponential_gaussian_pdf( x, p["q0_mV_ns"], p["sigma0_mV_ns"], p["alpha"]  ) )

    return { 
        "gaussian": ( p["n_total"] * bin_width * gaussian_part ),
        "expgaussian": ( p["n_total"] * bin_width * expgauss_part ),
     }


PEDESTAL_BELLAMY = FitModel(
    name="pedestal Bellamy",

    parameter_names=(
        "n_total",
        "q0_mV_ns",
        "sigma0_mV_ns",
        'w',
        'alpha'
    ),

    model=bellamy_pedestal_model,

    # components=bellamy_pedestal_components,

    pretty_parameter_names={
        "n_total": r"N_{\mathrm{tot}}",
        "q0_mV_ns": r"Q_0",
        "sigma0_mV_ns": r"\sigma_0",
        "w": r"w",
        "alpha": r"\alpha",
    },
)