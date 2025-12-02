from dataclasses import dataclass

import numpy as np

from .grid import Grid2D
from .lensing_operator import LensingOperator

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:

    @njit
    def _naive_reconstruct(beta_x_flat, beta_y_flat, intens_flat, bx_min, bx_max, by_min, by_max, src_n):
        n_pix2 = src_n * src_n
        source_flat = np.zeros(n_pix2, dtype=np.float64)
        counts_flat = np.zeros(n_pix2, dtype=np.float64)
        scale_x = (src_n - 1.0) / (bx_max - bx_min)
        scale_y = (src_n - 1.0) / (by_max - by_min)
        n = beta_x_flat.size
        for i in range(n):
            ix = int((beta_x_flat[i] - bx_min) * scale_x)
            iy = int((beta_y_flat[i] - by_min) * scale_y)
            if ix < 0:
                ix = 0
            elif ix >= src_n:
                ix = src_n - 1
            if iy < 0:
                iy = 0
            elif iy >= src_n:
                iy = src_n - 1
            idx = iy * src_n + ix
            source_flat[idx] += intens_flat[i]
            counts_flat[idx] += 1.0
        # Normalize where counts > 0
        for j in range(n_pix2):
            if counts_flat[j] > 0.0:
                source_flat[j] /= counts_flat[j]
        return source_flat.reshape((src_n, src_n))


@dataclass
class NaiveUnshooter:
    """Ray-unshooting inverse without explicit regularization."""

    image_grid: Grid2D
    source_grid: Grid2D
    beta_x: np.ndarray
    beta_y: np.ndarray

    def reconstruct(self, data_image: np.ndarray) -> np.ndarray:
        src_n = self.source_grid.n_pix
        x_src = self.source_grid.x
        y_src = self.source_grid.y

        bx_min, bx_max = x_src.min(), x_src.max()
        by_min, by_max = y_src.min(), y_src.max()

        beta_x_flat = self.beta_x.ravel()
        beta_y_flat = self.beta_y.ravel()
        intens_flat = data_image.ravel()

        if _HAS_NUMBA:
            return _naive_reconstruct(beta_x_flat, beta_y_flat, intens_flat, bx_min, bx_max, by_min, by_max, src_n)

        ix = ((beta_x_flat - bx_min) / (bx_max - bx_min) * (src_n - 1)).astype(int)
        iy = ((beta_y_flat - by_min) / (by_max - by_min) * (src_n - 1)).astype(int)
        ix = np.clip(ix, 0, src_n - 1)
        iy = np.clip(iy, 0, src_n - 1)

        source = np.zeros((src_n, src_n), dtype=float)
        counts = np.zeros_like(source)

        np.add.at(source, (iy, ix), intens_flat)
        np.add.at(counts, (iy, ix), 1.0)

        mask = counts > 0
        source[mask] /= counts[mask]
        return source


@dataclass
class TikhonovInversion:
    """Tikhonov inversion with optional regularizer."""

    operator: LensingOperator
    lambda_reg: float = 1e-2
    R: np.ndarray | None = None

    def reconstruct(self, data_image: np.ndarray) -> np.ndarray:
        n_src = self.operator.source_grid.n_pix
        N_src2 = n_src * n_src

        # ATd via gather
        d_flat = data_image.ravel()
        ATd = np.bincount(self.operator._src_indices, weights=d_flat, minlength=N_src2).astype(float)

        counts = self.operator._counts

        if _HAS_SCIPY and sp.issparse(self.R):
            RtR = self.R.T @ self.R
            diag_counts = sp.diags(counts)
            M = diag_counts + self.lambda_reg * RtR
            try:
                s_hat, info = spla.cg(M, ATd, tol=1e-6, maxiter=200)
                if info != 0:
                    s_hat = spla.spsolve(M, ATd)
            except TypeError:
                # Fallback for SciPy variants: directly solve
                s_hat = spla.spsolve(M, ATd)
        else:
            if self.R is None:
                M = np.diag(counts) + self.lambda_reg * np.eye(N_src2)
            else:
                RtR = self.R.T @ self.R
                M = np.diag(counts) + self.lambda_reg * RtR
            s_hat = np.linalg.solve(M, ATd)

        return s_hat.reshape(n_src, n_src)

    def model_from_data(self, data_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source = self.reconstruct(data_image)
        model = self.operator.forward(source)
        return source, model
