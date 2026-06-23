import numpy as np



def gaussian_pdf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)