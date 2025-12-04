from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from synthetic_lensing.gaussian_source import GaussianSource
from synthetic_lensing.light_sources import LightProfile
from synthetic_lensing.grid import Grid2D
from synthetic_lensing.lens_models import BaseLensModel
from synthetic_lensing.lensing_operator import LensingOperator
from synthetic_lensing.regularizers import build_laplacian_regularizer

try:
    import scipy.ndimage as ndi
except ImportError:  
    ndi = None


_GRID_CACHE: dict[tuple[int, float], Grid2D] = {}
_OP_CACHE: dict[tuple, LensingOperator] = {}
_REG_CACHE: dict[int, object] = {}


def _lens_key(n_pix: int, fov: float, lens: BaseLensModel) -> tuple:
    if hasattr(lens, "q"):
        return (
            lens.__class__.__name__,
            n_pix,
            round(float(fov), 6),
            round(float(lens.theta_E), 6),
            round(float(lens.x0), 6),
            round(float(lens.y0), 6),
            round(float(getattr(lens, "q", 1.0)), 6),
            round(float(getattr(lens, "phi_deg", 0.0)), 6),
        )
    return (
        lens.__class__.__name__,
        n_pix,
        round(float(fov), 6),
        round(float(lens.theta_E), 6),
        round(float(lens.x0), 6),
        round(float(lens.y0), 6),
    )


def get_grid(n_pix: int, fov: float) -> Grid2D:
    key = (n_pix, fov)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = Grid2D.from_fov(n_pix, fov)
    return _GRID_CACHE[key]


def get_operator(n_pix: int, fov: float, lens: BaseLensModel) -> LensingOperator:
    key = _lens_key(n_pix, fov, lens)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    grid = get_grid(n_pix, fov)
    op = LensingOperator(grid, grid, lens)
    _OP_CACHE[key] = op
    return op


def get_regularizer(n_pix: int):
    if n_pix not in _REG_CACHE:
        _REG_CACHE[n_pix] = build_laplacian_regularizer(n_pix)
    return _REG_CACHE[n_pix]


class SourceModel:
    """Interface for source models."""

    def evaluate(self, grid: Grid2D) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GaussianSourceModel(SourceModel):
    src: GaussianSource

    def evaluate(self, grid: Grid2D) -> np.ndarray:
        return self.src.evaluate(grid.X, grid.Y)


@dataclass
class LightProfileSourceModel(SourceModel):
    profile: LightProfile
    normalize: bool = False

    def evaluate(self, grid: Grid2D) -> np.ndarray:
        arr = self.profile.evaluate(grid.X, grid.Y)
        if self.normalize and np.max(arr) != 0:
            arr = arr / np.max(arr)
        return arr


@dataclass
class ImageSourceModel(SourceModel):
    image: np.ndarray
    normalize: bool = True

    def evaluate(self, grid: Grid2D) -> np.ndarray:
        arr = np.asarray(self.image, dtype=float)
        if arr.shape != grid.X.shape:
            if ndi is None:
                raise ValueError("Image shape does not match grid and scipy.ndimage is not available for resizing.")
            zoom_y = grid.n_pix / arr.shape[0]
            zoom_x = grid.n_pix / arr.shape[1]
            arr = ndi.zoom(arr, (zoom_y, zoom_x), order=1)
        if self.normalize and arr.max() != 0:
            arr = arr / arr.max()
        return arr


class Transform:
    """Base class for image-plane transforms."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NoiseTransform(Transform):
    sigma: float
    rng: Optional[np.random.Generator] = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.sigma <= 0:
            return image
        rng = self.rng if self.rng is not None else np.random.default_rng()
        noise = rng.normal(0.0, self.sigma, size=image.shape)
        return image + noise


@dataclass
class PoissonNoiseTransform(Transform):

    rng: Optional[np.random.Generator] = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        noisy = rng.poisson(np.clip(image, 0.0, None)).astype(float)
        return noisy


@dataclass
class NormalizeTransform(Transform):

    eps: float = 1e-8

    def apply(self, image: np.ndarray) -> np.ndarray:
        maxv = float(np.max(np.abs(image)))
        if maxv < self.eps:
            return image
        return image / maxv


@dataclass
class BlurTransform(Transform):
    sigma: float = 0.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.sigma <= 0 or ndi is None:
            return image
        return ndi.gaussian_filter(image, sigma=self.sigma)


@dataclass
class MaskTransform(Transform):
    """Apply a precomputed mask (multiply)."""

    mask: np.ndarray

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image * self.mask


@dataclass
class ForwardModel:

    n_pix: int
    fov: float
    source_model: SourceModel
    lens_model: BaseLensModel
    transforms: Optional[List[Transform]] = None

    def simulate(self) -> dict:
        grid = get_grid(self.n_pix, self.fov)
        op = get_operator(self.n_pix, self.fov, self.lens_model)

        source_true = self.source_model.evaluate(grid)
        image_clean = op.forward(source_true)

        image_obs = image_clean
        if self.transforms:
            for t in self.transforms:
                image_obs = t.apply(image_obs)

        return {
            "source_true": source_true,
            "image_clean": image_clean,
            "image_obs": image_obs,
            "operator": op,
            "grid": grid,
        }
