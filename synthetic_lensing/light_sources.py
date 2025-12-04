from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class LightProfile(ABC):

    @abstractmethod
    def evaluate(self, BX: np.ndarray, BY: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class EllipticalGaussianLight(LightProfile):
    I0: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    sigma_major: float = 0.4
    sigma_minor: float = 0.2
    phi_deg: float = 0.0

    def evaluate(self, BX: np.ndarray, BY: np.ndarray) -> np.ndarray:
        dx = BX - self.x0
        dy = BY - self.y0
        phi = np.deg2rad(self.phi_deg)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        x_p = cosphi * dx + sinphi * dy
        y_p = -sinphi * dx + cosphi * dy
        sig_maj = max(self.sigma_major, 1e-6)
        sig_min = max(self.sigma_minor, 1e-6)
        r2 = (x_p / sig_maj) ** 2 + (y_p / sig_min) ** 2
        return self.I0 * np.exp(-0.5 * r2)


@dataclass
class SersicLight(LightProfile):
    I0: float = 1.0
    x0: float = 0.0
    y0: float = 0.0
    r_eff: float = 0.5
    n: float = 2.0
    q: float = 1.0
    phi_deg: float = 0.0

    def evaluate(self, BX: np.ndarray, BY: np.ndarray) -> np.ndarray:
        dx = BX - self.x0
        dy = BY - self.y0
        phi = np.deg2rad(self.phi_deg)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        x_p = cosphi * dx + sinphi * dy
        y_p = -sinphi * dx + cosphi * dy
        q_clipped = np.clip(self.q, 0.1, 1.0)
        r_ell = np.sqrt(x_p**2 + (y_p / q_clipped) ** 2)
        r_eff = max(self.r_eff, 1e-6)
        n_ser = max(self.n, 0.1)
        bn = 1.9992 * n_ser - 0.3271  # approximate
        return self.I0 * np.exp(-bn * ((r_ell / r_eff) ** (1.0 / n_ser) - 1.0))
