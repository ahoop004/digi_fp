import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=image.shape)
    return image + noise
