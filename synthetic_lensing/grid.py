from dataclasses import dataclass

import numpy as np


@dataclass
class Grid2D:

    n_pix: int
    fov: float

    def __post_init__(self) -> None:
        half = self.fov / 2.0
        self.x = np.linspace(-half, half, self.n_pix)
        self.y = np.linspace(-half, half, self.n_pix)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

    @classmethod
    def from_fov(cls, n_pix: int, fov: float) -> "Grid2D":
        return cls(n_pix=n_pix, fov=fov)
