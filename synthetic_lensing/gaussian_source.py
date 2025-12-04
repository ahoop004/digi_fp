from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianSource:

    I0: float = 1.0
    beta0_x: float = 0.0
    beta0_y: float = 0.0
    sigma_major: float = 0.4
    sigma_minor: float = 0.2
    phi_deg: float = 0.0

    def evaluate(self, BX: np.ndarray, BY: np.ndarray) -> np.ndarray:

        dx = BX - self.beta0_x
        dy = BY - self.beta0_y

        phi = np.deg2rad(self.phi_deg)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)


        x_p = cosphi * dx + sinphi * dy
        y_p = -sinphi * dx + cosphi * dy

        sig_maj = max(self.sigma_major, 1e-6)
        sig_min = max(self.sigma_minor, 1e-6)

        r2 = (x_p / sig_maj) ** 2 + (y_p / sig_min) ** 2
        return self.I0 * np.exp(-0.5 * r2)
