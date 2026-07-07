import numpy as np
from scipy.special import erfc



def gaussian_pdf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

def exponential_gaussian_pdf(x, mean, sigma, alpha):
    """PDF of Gaussian(mean, sigma) convolved with alpha*exp(-alpha*x), x>=0."""
    sigma = np.maximum(sigma, 1e-12)
    alpha = np.maximum(alpha, 1e-12)
    z = (mean + alpha * sigma**2 - x) / (np.sqrt(2) * sigma)
    return 0.5 * alpha * np.exp(alpha * (mean - x) + 0.5 * (alpha * sigma) ** 2) * erfc(z)