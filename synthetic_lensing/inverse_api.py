from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from synthetic_lensing.inversion import NaiveUnshooter, TikhonovInversion
from synthetic_lensing.lensing_operator import LensingOperator


class InverseMethod(ABC):

    @abstractmethod
    def reconstruct(self, data_image: np.ndarray, operator: LensingOperator, regularizer=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Return (source, model_image)."""
        raise NotImplementedError


class NaiveInverse(InverseMethod):
    def reconstruct(self, data_image: np.ndarray, operator: LensingOperator, regularizer=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        beta_x, beta_y = operator.lens.map_to_source(operator.image_grid.X, operator.image_grid.Y)
        inv = NaiveUnshooter(operator.image_grid, operator.source_grid, beta_x, beta_y)
        source = inv.reconstruct(data_image)
        model = operator.forward(source)
        return source, model


@dataclass
class TikhonovInverse(InverseMethod):
    lambda_reg: float = 1e-2

    def reconstruct(self, data_image: np.ndarray, operator: LensingOperator, regularizer=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        inv = TikhonovInversion(operator, lambda_reg=self.lambda_reg, R=regularizer)
        source, model = inv.model_from_data(data_image)
        return source, model
