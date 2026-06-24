import numpy as np

from .core import FitModel
from .pdfs import gaussian_pdf
from .core import fit_histogram_model


def fit_pedestal( charge_mV_ns, p0, fit_range=None, bins=250, maxfev=100000 ):
    
    return fit_histogram_model(
        charge_mV_ns,
        model=PEDESTAL_GAUS,
        p0=p0,
        fit_range=fit_range,
        bins=bins,
        maxfev=maxfev,
    )


def pedestal_model( x, n_total, q0_mV_ns, sigma0_mV_ns, bin_width ):
    # Pedestal = Gaussian(Q0, σ0)
    return ( n_total * bin_width * gaussian_pdf( x, q0_mV_ns, sigma0_mV_ns, ) )


def pedestal_components( x, parameters, bin_width,):
    return { "pedestal": parameters["n_total"] * bin_width * gaussian_pdf( x, parameters["q0_mV_ns"], parameters["sigma0_mV_ns"] ) }


PEDESTAL_GAUS = FitModel(
    name="pedestal Gaussian",

    parameter_names=(
        "n_total",
        "q0_mV_ns",
        "sigma0_mV_ns",
    ),

    model=pedestal_model,

    components=pedestal_components,

    pretty_parameter_names={
        "n_total": r"N_{\mathrm{tot}}",
        "q0_mV_ns": r"Q_0",
        "sigma0_mV_ns": r"\sigma_0",
    },
)