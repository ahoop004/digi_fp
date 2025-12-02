## Synthetic lensing demo tasks

- [x] Create core grids module `grid.py` with `Grid2D` (including `from_fov`) providing reusable 2D coordinates/extents.
- [x] Implement analytic source model in `gaussian_source.py` with `GaussianSource.evaluate` (elliptical Gaussian with rotation and sigma clipping).
- [x] Add lens models in `lens_models.py`: base interface plus `SISLens` and `SIELens` deflection and `map_to_source`.
- [x] Build ray-tracing and operator in `lensing_operator.py`: `RayTracer` to compute beta fields and `LensingOperator` to assemble dense A matrix and forward mapping.
- [x] Add Laplacian regularizer builder `build_laplacian_regularizer` in `regularizers.py` (5-point stencil over source grid).
- [x] Implement inversions in `inversion.py`: `NaiveUnshooter` (ray-unshoot with accumulate/divide) and `TikhonovInversion` (normal equations with optional R).
- [x] Provide noise utilities in `noise.py` (Gaussian noise addition) and optional `metrics.py` (MSE/PSNR).
- [x] Verify pipeline non-interactively: script/notebook to run one example, compute MSE for naive vs. Tikhonov, and plot static 2×3 grid (true, lensed clean/noisy, naive recon, Tikhonov recon, residual).
- [x] Build interactive demo `synthetic_gui.py` using Plotly Dash with sliders for source, lens, noise, and λ; maintain six panels updating in callbacks.
- [x] Polish Dash UI: lens type selector, cleaner slider layout/marks, value readouts, and preset buttons/toggles for common setups.
- [x] Add run instructions in README (usage for static example and GUI).

Here’s a more detailed, implementation-ready TODO list you can hand to your coding agent. It’s structured by file, with concrete steps, suggested signatures, and sanity checks.

You can literally paste this into an issue/task list.

---

## 0. Project structure

Ask the agent to create a small package like:

```text
synthetic_lensing/
  __init__.py
  grid.py
  gaussian_source.py
  lens_models.py
  lensing_operator.py
  regularizers.py
  inversion.py
  noise.py
  metrics.py
  synthetic_demo_static.py   # non-interactive example
  synthetic_gui.py           # interactive slider demo
```

All modules should be importable both from scripts and from a Jupyter notebook.

---

## 1. `grid.py` – core 2D grid

**Goal:** Provide a reusable 2D coordinate grid.

### Tasks

* [x] Define `Grid2D` as a `@dataclass`.

  ```python
  from dataclasses import dataclass
  import numpy as np

  @dataclass
  class Grid2D:
      n_pix: int
      fov: float  # extent in same units as lens/source coordinates

      def __post_init__(self) -> None:
          half = self.fov / 2.0
          self.x = np.linspace(-half, half, self.n_pix)
          self.y = np.linspace(-half, half, self.n_pix)
          self.X, self.Y = np.meshgrid(self.x, self.y)
          self.extent = [self.x.min(), self.x.max(),
                         self.y.min(), self.y.max()]
  ```

* [x] Add a convenience constructor:

  ```python
  @classmethod
  def from_fov(cls, n_pix: int, fov: float) -> "Grid2D":
      return cls(n_pix=n_pix, fov=fov)
  ```

* [x] Add a minimal `__repr__` or rely on dataclass default.

**Sanity check:**

* Instantiate `Grid2D(64, 4.0)` in a notebook and verify:

  * `X.shape == (64, 64)`
  * `extent == [-2.0, 2.0, -2.0, 2.0]`.

---

## 2. `gaussian_source.py` – analytic source model

**Goal:** Implement a parameterized elliptical Gaussian.

### Tasks

* [x] Define `GaussianSource` dataclass:

  ```python
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
  ```

* [x] Implement `evaluate(self, BX, BY) -> np.ndarray`:

  * Inputs:

    * `BX`, `BY`: 2D arrays of same shape (from `Grid2D.X`, `Grid2D.Y`).
  * Steps:

    1. Compute offsets:

       ```python
       dx = BX - self.beta0_x
       dy = BY - self.beta0_y
       ```
    2. Convert `phi_deg` to radians.
    3. Rotate offsets into principal frame:

       ```python
       cosphi = np.cos(phi)
       sinphi = np.sin(phi)
       x_p =  cosphi * dx + sinphi * dy
       y_p = -sinphi * dx + cosphi * dy
       ```
    4. Enforce positive sigmas:

       ```python
       sig_maj = max(self.sigma_major, 1e-6)
       sig_min = max(self.sigma_minor, 1e-6)
       ```
    5. Compute radius and intensity:

       ```python
       r2 = (x_p / sig_maj)**2 + (y_p / sig_min)**2
       return self.I0 * np.exp(-0.5 * r2)
       ```

**Sanity check:**

* Create a `Grid2D`, evaluate a default `GaussianSource`, plot with `imshow`, verify shape and symmetry.

---

## 3. `lens_models.py` – SIS / SIE lenses

**Goal:** Implement simple lens mass models and deflection fields.

### Tasks

* [x] Define an abstract base class:

  ```python
  from abc import ABC, abstractmethod
  import numpy as np

  class BaseLensModel(ABC):
      @abstractmethod
      def deflection(self, theta_x, theta_y):
          """Return (alpha_x, alpha_y) for given theta_x, theta_y."""
          raise NotImplementedError

      def map_to_source(self, theta_x, theta_y):
          alpha_x, alpha_y = self.deflection(theta_x, theta_y)
          return theta_x - alpha_x, theta_y - alpha_y
  ```

* [x] Implement `SISLens`:

  ```python
  from dataclasses import dataclass

  @dataclass
  class SISLens(BaseLensModel):
      theta_E: float
      x0: float = 0.0
      y0: float = 0.0

      def deflection(self, theta_x, theta_y):
          dx = theta_x - self.x0
          dy = theta_y - self.y0
          r = np.hypot(dx, dy)
          eps = 1e-6
          r_safe = np.where(r == 0, eps, r)
          alpha_x = self.theta_E * dx / r_safe
          alpha_y = self.theta_E * dy / r_safe
          return alpha_x, alpha_y
  ```

* [x] Implement `SIELens` (approximate):

  ```python
  @dataclass
  class SIELens(BaseLensModel):
      theta_E: float
      x0: float = 0.0
      y0: float = 0.0
      q: float = 0.8
      phi_deg: float = 0.0

      def deflection(self, theta_x, theta_y):
          dx = theta_x - self.x0
          dy = theta_y - self.y0

          phi = np.deg2rad(self.phi_deg)
          cosphi, sinphi = np.cos(phi), np.sin(phi)

          x_p =  cosphi * dx + sinphi * dy
          y_p = -sinphi * dx + cosphi * dy

          q = np.clip(self.q, 0.1, 1.0)
          r_ell = np.sqrt(x_p**2 + (y_p / q)**2)
          eps = 1e-6
          r_safe = np.where(r_ell == 0, eps, r_ell)

          alpha_r = self.theta_E / np.sqrt(q)
          alpha_x_p = alpha_r * x_p / r_safe
          alpha_y_p = alpha_r * (y_p / q**2) / r_safe

          alpha_x =  cosphi * alpha_x_p - sinphi * alpha_y_p
          alpha_y =  sinphi * alpha_x_p + cosphi * alpha_y_p
          return alpha_x, alpha_y
  ```

**Sanity check:**

* For a circular SIS (`q=1`), check deflection is radial and magnitude ~constant at given radius.

---

## 4. `lensing_operator.py` – ray-tracer and A matrix

**Goal:** Build the mapping (d = A s) and provide `forward()`.

### Tasks

* [x] Import `Grid2D` and `BaseLensModel`.

* [x] Implement `RayTracer`:

  ```python
  from dataclasses import dataclass

  @dataclass
  class RayTracer:
      image_grid: Grid2D
      lens: BaseLensModel

      def beta(self):
          return self.lens.map_to_source(self.image_grid.X,
                                         self.image_grid.Y)
  ```

* [x] Implement `LensingOperator`:

  ```python
  @dataclass
  class LensingOperator:
      image_grid: Grid2D
      source_grid: Grid2D
      lens: BaseLensModel

      def __post_init__(self):
          self._build_index_map()
          self._build_matrix()
  ```

* [x] `_build_index_map`:

  Steps:

  1. Compute `beta_x, beta_y = lens.map_to_source(image_grid.X, image_grid.Y)`.
  2. Flatten `beta_x`, `beta_y`.
  3. Map to indices in source grid:

     * Get `x_src = source_grid.x`, `y_src = source_grid.y`.
     * `bx_min, bx_max = x_src.min(), x_src.max()`; same for y.
     * `ix = ((beta_x_flat - bx_min) / (bx_max - bx_min) * (n_src-1)).astype(int)`
     * `iy = ((beta_y_flat - by_min) / (by_max - by_min) * (n_src-1)).astype(int)`
     * Clip to `[0, n_src-1]`.
  4. Store `self._ix_src`, `self._iy_src`.

* [x] `_build_matrix`:

  Steps:

  1. `N_img2 = n_img * n_img`, `N_src2 = n_src * n_src`.
  2. Allocate `A = np.zeros((N_img2, N_src2))`.
  3. For each image index `i in range(N_img2)`:

     * `k = iy[i] * n_src + ix[i]`
     * `A[i, k] = 1.0`.
  4. Store `self.A`.

* [x] `forward(source_2d) -> image_2d`:

  ```python
  def forward(self, source_2d):
      s = source_2d.ravel()
      d = self.A @ s
      n_img = self.image_grid.n_pix
      return d.reshape(n_img, n_img)
  ```

* [x] `as_matrix()` returns `self.A`.

**Sanity check:**

* For a uniform source (all ones), `forward()` output should reflect the mapping (every image pixel has intensity equal to the source pixel it samples).

---

## 5. `regularizers.py` – Laplacian builder

**Goal:** Generate a curvature regularizer (R).

### Tasks

* [x] Implement `build_laplacian_regularizer(n_pix: int) -> np.ndarray`:

  * `N = n_pix * n_pix`
  * `R` shape `(N, N)` initialised to zeros.
  * Helper `idx(i, j) = i * n_pix + j`.
  * For each pixel `(i, j)`:

    * `k = idx(i, j)`
    * `R[k, k] = 4`
    * If left neighbor exists: `R[k, idx(i, j-1)] = -1`
    * If right neighbor exists: `R[k, idx(i, j+1)] = -1`
    * If up/down neighbors exist: set similarly.

**Sanity check:**

* For a constant vector `s` (all ones), `R @ s` should be near zero (up to boundary effects).

---

## 6. `inversion.py` – Naive and Tikhonov

**Goal:** Implement naive ray-unshooting and Tikhonov inversion.

### Tasks

* [x] Implement `NaiveUnshooter`:

  ```python
  @dataclass
  class NaiveUnshooter:
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

          ix = ((beta_x_flat - bx_min) / (bx_max - bx_min) * (src_n-1)).astype(int)
          iy = ((beta_y_flat - by_min) / (by_max - by_min) * (src_n-1)).astype(int)
          ix = np.clip(ix, 0, src_n-1)
          iy = np.clip(iy, 0, src_n-1)

          source = np.zeros((src_n, src_n), dtype=float)
          counts = np.zeros_like(source)

          np.add.at(source, (iy, ix), intens_flat)
          np.add.at(counts, (iy, ix), 1.0)

          mask = counts > 0
          source[mask] /= counts[mask]
          return source
  ```

* [x] Implement `TikhonovInversion`:

  ```python
  @dataclass
  class TikhonovInversion:
      operator: LensingOperator
      lambda_reg: float = 1e-2
      R: np.ndarray | None = None

      def reconstruct(self, data_image: np.ndarray) -> np.ndarray:
          A = self.operator.as_matrix()
          d = data_image.ravel()
          N_src2 = A.shape[1]

          ATA = A.T @ A
          ATd = A.T @ d

          if self.R is None:
              M = ATA + self.lambda_reg * np.eye(N_src2)
          else:
              RtR = self.R.T @ self.R
              M = ATA + self.lambda_reg * RtR

          s_hat = np.linalg.solve(M, ATd)
          n_src = self.operator.source_grid.n_pix
          return s_hat.reshape(n_src, n_src)

      def model_from_data(self, data_image: np.ndarray):
          src = self.reconstruct(data_image)
          img_model = self.operator.forward(src)
          return src, img_model
  ```

**Sanity check:**

* Use a tiny grid (e.g., 16×16) and run inversion on synthetic data; ensure code runs without singular-matrix errors for reasonable λ.

---

## 7. `noise.py` – Gaussian noise

**Goal:** Simple additive Gaussian noise.

### Tasks

* [x] Implement:

  ```python
  import numpy as np

  def add_gaussian_noise(image: np.ndarray, sigma: float, rng=None) -> np.ndarray:
      if rng is None:
          rng = np.random.default_rng()
      noise = rng.normal(0.0, sigma, size=image.shape)
      return image + noise
  ```

---

## 8. `metrics.py` – MSE/PSNR

**Goal:** Provide metrics for comparing reconstructions.

### Tasks

* [x] Implement:

  ```python
  import numpy as np

  def mse(a, b) -> float:
      a = np.asarray(a, dtype=float)
      b = np.asarray(b, dtype=float)
      return float(np.mean((a - b)**2))

  def psnr(a, b, data_range=None) -> float:
      a = np.asarray(a, dtype=float)
      b = np.asarray(b, dtype=float)
      if data_range is None:
          data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
      m = mse(a, b)
      if m == 0:
          return float("inf")
      return float(10.0 * np.log10((data_range**2) / m))
  ```

---

## 9. `synthetic_demo_static.py` – non-interactive test

**Goal:** Verify the whole pipeline once, with static plots and metrics.

### Tasks

* [x] In this script:

  1. Create grids:

     ```python
     img_grid = Grid2D.from_fov(64, 4.0)
     src_grid = Grid2D.from_fov(64, 4.0)
     ```

  2. Instantiate a `GaussianSource` with some parameters.

  3. Evaluate `source_true = gaussian.evaluate(src_grid.X, src_grid.Y)`.

  4. Instantiate a lens (`SIELens` or `SISLens`) with reasonable parameters.

  5. Build `LensingOperator(img_grid, src_grid, lens)`.

  6. Compute `image_clean = op.forward(source_true)`.

  7. Add noise: `image_noisy = add_gaussian_noise(image_clean, sigma=0.02*image_clean.max())`.

  8. Ray-trace: `BETA_X, BETA_Y = lens.map_to_source(img_grid.X, img_grid.Y)`.

  9. Naive recon: `source_naive = NaiveUnshooter(img_grid, src_grid, BETA_X, BETA_Y).reconstruct(image_noisy)`.

  10. Build `R = build_laplacian_regularizer(src_grid.n_pix)`.

  11. Tikhonov recon:

      ```python
      tik = TikhonovInversion(op, lambda_reg=1e-2, R=R)
      source_tik, image_model = tik.model_from_data(image_noisy)
      ```

  12. Compute metrics:

      * `mse_naive = mse(source_true, source_naive)`
      * `mse_tik = mse(source_true, source_tik)`

  13. Plot a 2×3 grid:

      * true source
      * lensed clean
      * lensed noisy
      * naive recon
      * Tikhonov recon
      * residual (image_noisy − image_model)

**Sanity check:**

* Confirm `mse_tik < mse_naive` for reasonable λ.

---

## 10. `synthetic_gui.py` – interactive demo (Plotly Dash)

**Goal:** Interactive sliders (web UI) to explore source, lens, noise, λ and see their effect on forward and inverse.

### Tasks

* [x] Dependencies: plotly, dash, numpy; import app modules (`Grid2D`, `GaussianSource`, `SIELens/SISLens`, `LensingOperator`, `build_laplacian_regularizer`, `NaiveUnshooter`, `TikhonovInversion`, `add_gaussian_noise`).

* [x] App scaffold: create Dash app with layout containing six `dcc.Graph` for the panels and grouped sliders (`dcc.Slider` or `dcc.Input`).

* [x] Initial data: build grids, default source/lens, Laplacian `R`; compute source_true, lensed clean/noisy, naive recon, Tikhonov recon, residual.

* [x] Callbacks: a single `@app.callback` taking all slider inputs to:
  - Update source and lens params; rebuild `LensingOperator` if lens changed.
  - Recompute forward, noise, naive recon, Tikhonov recon, residual.
  - Return updated figures for the six panels (use `imshow`-style heatmaps via `px.imshow` or `go.Heatmap` with shared color scales).

* [x] Run: `if __name__ == "__main__": app.run_server(debug=True)`; document usage (`python -m synthetic_lensing.synthetic_gui` or `python synthetic_lensing/synthetic_gui.py`).

---

## 11. Dash UI polish

**Goal:** Improve usability/clarity of the Dash app.

### Tasks

* [x] Add lens-type selector (dropdown for SIS/SIE) and conditionally use `SISLens` defaults (`q=1, phi=0`) when selected.
* [x] Clean up control layout: group sliders into a sidebar or tidy grid, add concise labels, reduce marks to endpoints/midpoints, enable tooltips to avoid tick clutter.
* [x] Add value readouts/badges next to key sliders (especially log10 λ showing λ).
* [x] Add preset buttons for common configurations (e.g., centered source, offset source, round lens, elliptical lens) and wire them to update controls.
* [x] Add optional toggles: freeze noise seed for repeatability and switch residual display (image-plane vs. source-plane).
* [x] Speed polish: cache/reuse lensing operator when lens unchanged, offer low-res/hi-res toggle, and add slider debounce on less-critical controls.

**Sanity check:**

* Moving source sliders morphs the true source and arcs.
* Moving lens sliders changes arc shape/pattern.
* Increasing noise makes reconstructions worse.
* Changing λ changes smoothness of Tikhonov source.

---

## 12. README – usage instructions

**Goal:** Short readme explaining:

* [x] How to run static demo:

  ```bash
  python synthetic_demo_static.py
  ```

* [x] How to run GUI:

  ```bash
  python synthetic_gui.py
  ```

* [x] What each slider does in the GUI.

---

You can hand this entire checklist to your coding agent as “implementation spec.” It’s explicit per-module, includes expected behavior and sanity checks, and keeps the structure aligned with your existing lensing + inversion work.
