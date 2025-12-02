# digi_fp

Synthetic lensing demo with forward modeling and regularized inversion.

## Running the static demo

```bash
python -m synthetic_lensing.synthetic_demo_static
```

This renders a 2×3 grid: true source, lensed clean/noisy images, naive and Tikhonov reconstructions, and residuals. MSE is shown in the titles.

## Running the interactive Dash app

```bash
python -m synthetic_lensing.synthetic_gui
```

Open the printed local URL in a browser. Controls:

- Source: intensity, position (βx, βy), widths (σ_maj/σ_min), rotation φ_S, presets (centered/offset).
- Lens: SIS/SIE selector, θ_E, center (x0, y0), axis ratio q and φ_L, presets (round/elliptical).
- Noise/regularization: resolution (48/64/96), residual mode (image/source), compute toggles (Naive/Tikhonov), noise σ, log10 λ, freeze-noise seed.

Notes:

- High resolution (96) is heavier; performance is improved via cached operators, Numba (if installed), and sparse solves when SciPy is available.
- Presets only apply when clicked; sliders remain live for fine tuning.
