import numpy as np


def mse(a, b) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.mean((a_arr - b_arr) ** 2))


def psnr(a, b, data_range=None) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if data_range is None:
        data_range = float(max(a_arr.max(), b_arr.max()) - min(a_arr.min(), b_arr.min()))
    m = mse(a_arr, b_arr)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / m))
