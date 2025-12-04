from dataclasses import dataclass

import numpy as np

from .grid import Grid2D
from .lens_models import BaseLensModel

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:

    @njit
    def _compute_indices(beta_x_flat, beta_y_flat, bx_min, bx_max, by_min, by_max, n_src):
        n = beta_x_flat.size
        src_indices = np.empty(n, dtype=np.int64)
        ix_arr = np.empty(n, dtype=np.int64)
        iy_arr = np.empty(n, dtype=np.int64)
        scale_x = (n_src - 1.0) / (bx_max - bx_min)
        scale_y = (n_src - 1.0) / (by_max - by_min)
        for i in range(n):
            ix = int((beta_x_flat[i] - bx_min) * scale_x)
            iy = int((beta_y_flat[i] - by_min) * scale_y)
            if ix < 0:
                ix = 0
            elif ix >= n_src:
                ix = n_src - 1
            if iy < 0:
                iy = 0
            elif iy >= n_src:
                iy = n_src - 1
            ix_arr[i] = ix
            iy_arr[i] = iy
            src_indices[i] = iy * n_src + ix
        return src_indices, ix_arr, iy_arr

    @njit
    def _gather_forward(src_flat, src_indices, n_img):
        out = np.empty(n_img * n_img, dtype=src_flat.dtype)
        for i in range(src_indices.size):
            out[i] = src_flat[src_indices[i]]
        return out.reshape((n_img, n_img))


# @dataclass
# class RayTracer:


#     image_grid: Grid2D
#     lens: BaseLensModel

#     def beta(self) -> tuple[np.ndarray, np.ndarray]:
#         return self.lens.map_to_source(self.image_grid.X, self.image_grid.Y)


@dataclass
class LensingOperator:


    image_grid: Grid2D
    source_grid: Grid2D
    lens: BaseLensModel

    def __post_init__(self) -> None:
        self._build_index_map()
        self.A = None

    def _build_index_map(self) -> None:
        beta_x, beta_y = self.lens.map_to_source(self.image_grid.X, self.image_grid.Y)
        beta_x_flat = beta_x.ravel()
        beta_y_flat = beta_y.ravel()

        x_src = self.source_grid.x
        y_src = self.source_grid.y

        bx_min, bx_max = x_src.min(), x_src.max()
        by_min, by_max = y_src.min(), y_src.max()

        n_src = self.source_grid.n_pix
        if _HAS_NUMBA:
            src_idx, ix_arr, iy_arr = _compute_indices(beta_x_flat, beta_y_flat, bx_min, bx_max, by_min, by_max, n_src)
            self._src_indices = src_idx
            self._ix_src = ix_arr
            self._iy_src = iy_arr
        else:
            ix = ((beta_x_flat - bx_min) / (bx_max - bx_min) * (n_src - 1)).astype(int)
            iy = ((beta_y_flat - by_min) / (by_max - by_min) * (n_src - 1)).astype(int)
            self._ix_src = np.clip(ix, 0, n_src - 1)
            self._iy_src = np.clip(iy, 0, n_src - 1)
            self._src_indices = self._iy_src * n_src + self._ix_src
            
        self._counts = np.bincount(self._src_indices, minlength=n_src * n_src).astype(float)

    def _build_matrix(self) -> None:
        n_img = self.image_grid.n_pix
        n_src = self.source_grid.n_pix
        n_img2 = n_img * n_img
        n_src2 = n_src * n_src

        A = np.zeros((n_img2, n_src2), dtype=float)
        A[np.arange(n_img2), self._src_indices] = 1.0
        self.A = A

    def forward(self, source_2d: np.ndarray) -> np.ndarray:
        s = source_2d.ravel()
        n_img = self.image_grid.n_pix
        if hasattr(self, "_src_indices"):
            if _HAS_NUMBA:
                return _gather_forward(s, self._src_indices, n_img)
            return s[self._src_indices].reshape(n_img, n_img)
        d = self.A @ s
        return d.reshape(n_img, n_img)

    def as_matrix(self) -> np.ndarray:
        if self.A is None:
            self._build_matrix()
        return self.A
