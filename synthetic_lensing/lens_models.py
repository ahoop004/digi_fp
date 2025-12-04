from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseLensModel(ABC):


    @abstractmethod
    def deflection(self, theta_x: np.ndarray, theta_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return deflection components (alpha_x, alpha_y)."""
        raise NotImplementedError

    def map_to_source(self, theta_x: np.ndarray, theta_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map image-plane angles to source-plane coordinates."""
        alpha_x, alpha_y = self.deflection(theta_x, theta_y)
        return theta_x - alpha_x, theta_y - alpha_y


@dataclass
class SISLens(BaseLensModel):
    """Singular isothermal sphere lens model."""

    theta_E: float
    x0: float = 0.0
    y0: float = 0.0

    def deflection(self, theta_x: np.ndarray, theta_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = theta_x - self.x0
        dy = theta_y - self.y0
        r = np.hypot(dx, dy)
        eps = 1e-6
        r_safe = np.where(r == 0, eps, r)
        alpha_x = self.theta_E * dx / r_safe
        alpha_y = self.theta_E * dy / r_safe
        return alpha_x, alpha_y


@dataclass
class SIELens(BaseLensModel):
    """Approximate singular isothermal ellipsoid lens model."""

    theta_E: float
    x0: float = 0.0
    y0: float = 0.0
    q: float = 0.8
    phi_deg: float = 0.0

    def deflection(self, theta_x: np.ndarray, theta_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = theta_x - self.x0
        dy = theta_y - self.y0

        phi = np.deg2rad(self.phi_deg)
        cosphi, sinphi = np.cos(phi), np.sin(phi)


        x_p = cosphi * dx + sinphi * dy
        y_p = -sinphi * dx + cosphi * dy

        q_clipped = np.clip(self.q, 0.1, 1.0)
        r_ell = np.sqrt(x_p**2 + (y_p / q_clipped) ** 2)
        eps = 1e-6
        r_safe = np.where(r_ell == 0, eps, r_ell)

        alpha_r = self.theta_E / np.sqrt(q_clipped)
        alpha_x_p = alpha_r * x_p / r_safe
        alpha_y_p = alpha_r * (y_p / q_clipped**2) / r_safe


        alpha_x = cosphi * alpha_x_p - sinphi * alpha_y_p
        alpha_y = sinphi * alpha_x_p + cosphi * alpha_y_p
        return alpha_x, alpha_y
