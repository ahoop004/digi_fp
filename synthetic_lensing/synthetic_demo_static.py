from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script (python synthetic_lensing/synthetic_demo_static.py)
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from synthetic_lensing.gaussian_source import GaussianSource
from synthetic_lensing.grid import Grid2D
from synthetic_lensing.inversion import NaiveUnshooter, TikhonovInversion
from synthetic_lensing.lens_models import SIELens
from synthetic_lensing.lensing_operator import LensingOperator
from synthetic_lensing.metrics import mse
from synthetic_lensing.noise import add_gaussian_noise
from synthetic_lensing.regularizers import build_laplacian_regularizer


def run_demo(n_pix: int = 64, fov: float = 4.0, noise_frac: float = 0.02, lambda_reg: float = 1e-2):
    img_grid = Grid2D.from_fov(n_pix, fov)
    src_grid = Grid2D.from_fov(n_pix, fov)

    source = GaussianSource(I0=1.0, beta0_x=0.0, beta0_y=0.0, sigma_major=0.4, sigma_minor=0.2, phi_deg=20.0)
    lens = SIELens(theta_E=1.4, x0=0.05, y0=-0.05, q=0.8, phi_deg=15.0)

    source_true = source.evaluate(src_grid.X, src_grid.Y)

    op = LensingOperator(img_grid, src_grid, lens)
    image_clean = op.forward(source_true)

    noise_sigma = noise_frac * image_clean.max()
    image_noisy = add_gaussian_noise(image_clean, sigma=noise_sigma)

    beta_x, beta_y = lens.map_to_source(img_grid.X, img_grid.Y)
    naive = NaiveUnshooter(img_grid, src_grid, beta_x, beta_y)
    source_naive = naive.reconstruct(image_noisy)

    R = build_laplacian_regularizer(src_grid.n_pix)
    tik = TikhonovInversion(op, lambda_reg=lambda_reg, R=R)
    source_tik, image_model = tik.model_from_data(image_noisy)

    mse_naive = mse(source_true, source_naive)
    mse_tik = mse(source_true, source_tik)

    residual = image_noisy - image_model

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    ax = axes.ravel()
    ims = []
    ims.append(ax[0].imshow(source_true, origin="lower", extent=src_grid.extent))
    ax[0].set_title("True source")
    ims.append(ax[1].imshow(image_clean, origin="lower", extent=img_grid.extent))
    ax[1].set_title("Lensed clean")
    ims.append(ax[2].imshow(image_noisy, origin="lower", extent=img_grid.extent))
    ax[2].set_title("Lensed noisy")
    ims.append(ax[3].imshow(source_naive, origin="lower", extent=src_grid.extent))
    ax[3].set_title(f"Naive recon (MSE={mse_naive:.3e})")
    ims.append(ax[4].imshow(source_tik, origin="lower", extent=src_grid.extent))
    ax[4].set_title(f"Tikhonov recon (MSE={mse_tik:.3e})")
    ims.append(ax[5].imshow(residual, origin="lower", extent=img_grid.extent))
    ax[5].set_title("Residual (data - model)")
    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
