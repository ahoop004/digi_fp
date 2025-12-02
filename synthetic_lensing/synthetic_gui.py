from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

# Allow running as a script: python synthetic_lensing/synthetic_gui.py
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from synthetic_lensing.gaussian_source import GaussianSource
from synthetic_lensing.grid import Grid2D
from synthetic_lensing.inversion import NaiveUnshooter, TikhonovInversion
from synthetic_lensing.lens_models import SIELens, SISLens
from synthetic_lensing.lensing_operator import LensingOperator
from synthetic_lensing.noise import add_gaussian_noise
from synthetic_lensing.regularizers import build_laplacian_regularizer


N_PIX_DEFAULT = 48
FOV = 4.0
_GRID_CACHE: dict[tuple[int, float], Grid2D] = {}
_REG_CACHE: dict[int, np.ndarray] = {}
_OP_CACHE: dict[tuple, LensingOperator] = {}
RNG = np.random.default_rng()


def get_grids(n_pix: int, fov: float = FOV) -> tuple[Grid2D, Grid2D]:
    key = (n_pix, fov)
    if key not in _GRID_CACHE:
        grid = Grid2D.from_fov(n_pix, fov)
        _GRID_CACHE[key] = grid
    grid = _GRID_CACHE[key]
    return grid, grid  # same grid for src and img here


def get_regularizer(n_pix: int) -> np.ndarray:
    if n_pix not in _REG_CACHE:
        _REG_CACHE[n_pix] = build_laplacian_regularizer(n_pix)
    return _REG_CACHE[n_pix]


def _lens_key(n_pix: int, lens) -> tuple:
    if isinstance(lens, SISLens):
        return ("SIS", n_pix, round(lens.theta_E, 6), round(lens.x0, 6), round(lens.y0, 6))
    if isinstance(lens, SIELens):
        return (
            "SIE",
            n_pix,
            round(lens.theta_E, 6),
            round(lens.x0, 6),
            round(lens.y0, 6),
            round(lens.q, 6),
            round(lens.phi_deg, 6),
        )
    return ("unknown", n_pix)


def get_operator(n_pix: int, lens) -> LensingOperator:
    key = _lens_key(n_pix, lens)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    src_grid, img_grid = get_grids(n_pix)
    op = LensingOperator(img_grid, src_grid, lens)
    _OP_CACHE[key] = op
    return op


def build_pipeline(
    n_pix: int,
    source: GaussianSource,
    lens: SIELens,
    noise_sigma: float,
    lambda_reg: float,
    freeze_noise: bool,
):
    src_grid, img_grid = get_grids(n_pix)
    source_true = source.evaluate(src_grid.X, src_grid.Y)

    op = get_operator(n_pix, lens)
    image_clean = op.forward(source_true)
    image_noisy = add_gaussian_noise(image_clean, sigma=noise_sigma, rng=RNG if freeze_noise else None)

    beta_x, beta_y = lens.map_to_source(img_grid.X, img_grid.Y)
    naive = NaiveUnshooter(img_grid, src_grid, beta_x, beta_y)
    source_naive = naive.reconstruct(image_noisy)
    image_model_naive = op.forward(source_naive)

    R_reg = get_regularizer(n_pix)
    tik = TikhonovInversion(op, lambda_reg=lambda_reg, R=R_reg)
    source_tik, image_model_tik = tik.model_from_data(image_noisy)

    return {
        "source_true": source_true,
        "image_clean": image_clean,
        "image_noisy": image_noisy,
        "source_naive": source_naive,
        "source_tik": source_tik,
        "image_model_naive": image_model_naive,
        "image_model_tik": image_model_tik,
        "src_grid": src_grid,
        "img_grid": img_grid,
    }


def heatmap(arr: np.ndarray, title: str, grid: Grid2D, showscale: bool = False) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=grid.x,
            y=grid.y,
            colorscale="Viridis",
            showscale=showscale,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=30, r=10, t=40, b=30),
        xaxis_title="x",
        yaxis_title="y",
        height=320,
    )
    fig.update_yaxes(autorange="reversed")  # emulate imshow origin="lower"
    return fig


def empty_heatmap(title: str, grid: Grid2D) -> go.Figure:
    zeros = np.zeros_like(grid.X)
    return heatmap(zeros, title, grid, showscale=False)


def simple_marks(lo: float, hi: float) -> dict:
    mid = (lo + hi) / 2.0
    return {lo: f"{lo:g}", mid: f"{mid:g}", hi: f"{hi:g}"}


def slider_row(
    id_: str,
    label: str,
    lo: float,
    hi: float,
    step: float,
    value: float,
    marks=None,
    tooltip=None,
    updatemode: str = "mouseup",
) -> html.Div:
    slider = dcc.Slider(
        id=id_,
        min=lo,
        max=hi,
        step=step,
        value=value,
        marks=marks if marks is not None else {lo: f"{lo:g}", hi: f"{hi:g}"},
        tooltip=tooltip or {"placement": "bottom"},
        updatemode=updatemode,
    )
    return html.Div(
        [
            html.Div(label, style={"width": "80px"}),
            slider,
            html.Div(id=f"{id_}-val", style={"width": "80px", "textAlign": "right", "fontSize": "12px"}),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "90px 1fr 70px",
            "alignItems": "center",
            "gap": "6px",
            "marginBottom": "8px",
        },
    )


def make_app() -> Dash:
    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H2("Synthetic Gravitational Lensing (Dash)"),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="true-source"),
                            dcc.Graph(id="lensed-clean"),
                            dcc.Graph(id="lensed-noisy"),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="naive-recon"),
                            dcc.Graph(id="tik-recon"),
                            dcc.Graph(id="residual"),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(3, 1fr)",
                            "gap": "12px",
                            "marginTop": "12px",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Source parameters"),
                            html.Div(
                                [
                                    html.Button("Centered source", id="preset-src-center", n_clicks=0, className="preset-btn"),
                                    html.Button("Offset source", id="preset-src-offset", n_clicks=0, className="preset-btn"),
                                ],
                                style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "8px"},
                            ),
                            slider_row("i0", "I0", 0.1, 2.0, 0.01, 1.0),
                            slider_row("bx", "βx", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
                            slider_row("by", "βy", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
                            slider_row("smaj", "σ_maj", 0.05, 1.0, 0.01, 0.4),
                            slider_row("smin", "σ_min", 0.05, 1.0, 0.01, 0.2),
                            slider_row("sphi", "φ_S", -90, 90, 1, 20, marks={-90: "-90", 0: "0", 90: "90"}),
                        ],
                        style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [
                            html.H4("Lens parameters (SIE)"),
                            html.Div(
                                [
                                    html.Div("Lens model", style={"width": "90px"}),
                                    dcc.Dropdown(
                                        id="lens-type",
                                        options=[
                                            {"label": "SIS", "value": "sis"},
                                            {"label": "SIE", "value": "sie"},
                                        ],
                                        value="sie",
                                        clearable=False,
                                        style={"width": "120px"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "90px 1fr",
                                    "alignItems": "center",
                                    "gap": "6px",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Button("Round lens", id="preset-lens-round", n_clicks=0, className="preset-btn"),
                                    html.Button("Elliptical lens", id="preset-lens-ellip", n_clicks=0, className="preset-btn"),
                                ],
                                style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "8px"},
                            ),
                            slider_row("thetaE", "θ_E", 0.5, 2.5, 0.01, 1.2),
                            slider_row("lx", "x0", -0.5, 0.5, 0.01, 0.0, marks=simple_marks(-0.5, 0.5)),
                            slider_row("ly", "y0", -0.5, 0.5, 0.01, 0.0, marks=simple_marks(-0.5, 0.5)),
                            slider_row("q", "q", 0.5, 1.0, 0.01, 0.85, marks={0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}),
                            slider_row("lphi", "φ_L", -45, 45, 1, 10, marks={-45: "-45", 0: "0", 45: "45"}),
                        ],
                        style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [
                            html.H4("Noise and regularization"),
                            html.Div(
                                [
                                    html.Div("Resolution", style={"width": "90px"}),
                                    dcc.RadioItems(
                                        id="resolution",
                                        options=[
                                            {"label": "Low (48)", "value": 48},
                                            {"label": "Med (64)", "value": 64},
                                            {"label": "High (96)", "value": 96},
                                        ],
                                        value=N_PIX_DEFAULT,
                                        inline=True,
                                        labelStyle={"marginRight": "10px"},
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "90px 1fr",
                                    "alignItems": "center",
                                    "gap": "6px",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(
                                [
                                    dcc.RadioItems(
                                        id="residual-mode",
                                        options=[
                                            {"label": "Residual: image plane", "value": "image"},
                                            {"label": "Residual: source plane", "value": "source"},
                                        ],
                                        value="image",
                                        inline=False,
                                        style={"fontSize": "13px"},
                                    )
                                ],
                                style={"marginBottom": "8px"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="recon-modes",
                                        options=[
                                            {"label": "Compute Naive", "value": "naive"},
                                            {"label": "Compute Tikhonov", "value": "tik"},
                                        ],
                                        value=["naive", "tik"],
                                        style={"fontSize": "13px"},
                                    )
                                ],
                                style={"marginBottom": "8px"},
                            ),
                            slider_row("noise", "σ_noise", 0.0, 0.1, 0.001, 0.02, marks={0.0: "0", 0.05: "0.05", 0.1: "0.1"}),
                            slider_row(
                                "lambda_log",
                                "log10 λ",
                                -4,
                                0,
                                0.1,
                                -2,
                                marks={-4: "1e-4", -3: "1e-3", -2: "1e-2", -1: "1e-1", 0: "1"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="freeze-noise",
                                        options=[{"label": "Freeze noise seed", "value": "freeze"}],
                                        value=[],
                                        style={"fontSize": "13px"},
                                    )
                                ],
                                style={"marginTop": "8px"},
                            ),
                        ],
                        style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "8px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(340px, 1fr))",
                    "gap": "12px",
                    "marginTop": "16px",
                },
            ),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
    )

    @app.callback(
        Output("bx", "value"),
        Output("by", "value"),
        Output("sphi", "value"),
        Output("lens-type", "value"),
        Output("q", "value"),
        Output("lphi", "value"),
        Input("preset-src-center", "n_clicks"),
        Input("preset-src-offset", "n_clicks"),
        Input("preset-lens-round", "n_clicks"),
        Input("preset-lens-ellip", "n_clicks"),
        State("bx", "value"),
        State("by", "value"),
        State("sphi", "value"),
        State("lens-type", "value"),
        State("q", "value"),
        State("lphi", "value"),
    )
    def apply_presets(n_center, n_offset, n_round, n_ellip, bx, by, sphi, lens_type, q, lphi):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trig == "preset-src-center":
            bx, by, sphi = 0.0, 0.0, 0.0
        elif trig == "preset-src-offset":
            bx, by, sphi = 0.6, -0.3, 20.0
        elif trig == "preset-lens-round":
            lens_type, q, lphi = "sis", 1.0, 0.0
        elif trig == "preset-lens-ellip":
            lens_type, q, lphi = "sie", 0.7, 20.0
        return bx, by, sphi, lens_type, q, lphi

    @app.callback(
        Output("true-source", "figure"),
        Output("lensed-clean", "figure"),
        Output("lensed-noisy", "figure"),
        Output("naive-recon", "figure"),
        Output("tik-recon", "figure"),
        Output("residual", "figure"),
        Output("i0-val", "children"),
        Output("bx-val", "children"),
        Output("by-val", "children"),
        Output("smaj-val", "children"),
        Output("smin-val", "children"),
        Output("sphi-val", "children"),
        Output("thetaE-val", "children"),
        Output("lx-val", "children"),
        Output("ly-val", "children"),
        Output("q-val", "children"),
        Output("lphi-val", "children"),
        Output("noise-val", "children"),
        Output("lambda_log-val", "children"),
        Input("freeze-noise", "value"),
        Input("lens-type", "value"),
        Input("resolution", "value"),
        Input("residual-mode", "value"),
        Input("recon-modes", "value"),
        Input("preset-src-center", "n_clicks"),
        Input("preset-src-offset", "n_clicks"),
        Input("preset-lens-round", "n_clicks"),
        Input("preset-lens-ellip", "n_clicks"),
        Input("i0", "value"),
        Input("bx", "value"),
        Input("by", "value"),
        Input("smaj", "value"),
        Input("smin", "value"),
        Input("sphi", "value"),
        Input("thetaE", "value"),
        Input("lx", "value"),
        Input("ly", "value"),
        Input("q", "value"),
        Input("lphi", "value"),
        Input("noise", "value"),
        Input("lambda_log", "value"),
    )
    def update_figs(
        freeze_noise_vals,
        lens_type,
        n_pix,
        residual_mode,
        recon_modes,
        n_center,
        n_offset,
        n_round,
        n_ellip,
        i0,
        bx,
        by,
        smaj,
        smin,
        sphi,
        thetaE,
        lx,
        ly,
        q,
        lphi,
        noise_sigma,
        lambda_log,
    ):
        # Normalize inputs
        recon_modes = recon_modes or []
        n_pix = int(n_pix)

        # Apply preset only if a preset button triggered this callback
        trig_id = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trig_id in {"preset-src-center", "preset-src-offset", "preset-lens-round", "preset-lens-ellip"}:
            if trig_id == "preset-src-center":
                bx, by, sphi = 0.0, 0.0, 0.0
            elif trig_id == "preset-src-offset":
                bx, by, sphi = 0.6, -0.3, 20.0
            elif trig_id == "preset-lens-round":
                lens_type = "sis"
                q, lphi = 1.0, 0.0
            elif trig_id == "preset-lens-ellip":
                lens_type = "sie"
                q, lphi = 0.7, 20.0

        source = GaussianSource(I0=i0, beta0_x=bx, beta0_y=by, sigma_major=smaj, sigma_minor=smin, phi_deg=sphi)
        if lens_type == "sis":
            lens = SISLens(theta_E=thetaE, x0=lx, y0=ly)
            q_eff, lphi_eff = 1.0, 0.0
        else:
            lens = SIELens(theta_E=thetaE, x0=lx, y0=ly, q=q, phi_deg=lphi)
            q_eff, lphi_eff = q, lphi
        lambda_reg = 10 ** lambda_log
        freeze_noise = "freeze" in (freeze_noise_vals or [])
        compute_naive = "naive" in (recon_modes or [])
        compute_tik = "tik" in (recon_modes or [])

        data = build_pipeline(n_pix, source, lens, noise_sigma, lambda_reg, freeze_noise)

        # Residual selection
        if residual_mode == "source":
            residual_fig = heatmap(data["source_true"] - data["source_tik"], "Residual (source true - recon)", data["src_grid"], showscale=True)
        else:
            model_img = data["image_model_tik"] if compute_tik else data["image_model_naive"]
            residual_fig = heatmap(data["image_noisy"] - model_img, "Residual (data - model)", data["img_grid"], showscale=True)

        return (
            heatmap(data["source_true"], "True source", data["src_grid"]),
            heatmap(data["image_clean"], "Lensed clean", data["img_grid"]),
            heatmap(data["image_noisy"], "Lensed noisy", data["img_grid"]),
            heatmap(data["source_naive"], "Naive recon" if compute_naive else "Naive recon (hidden)", data["src_grid"])
            if compute_naive
            else empty_heatmap("Naive recon (hidden)", data["src_grid"]),
            heatmap(data["source_tik"], "Tikhonov recon" if compute_tik else "Tikhonov recon (hidden)", data["src_grid"])
            if compute_tik
            else empty_heatmap("Tikhonov recon (hidden)", data["src_grid"]),
            residual_fig,
            f"{i0:.2f}",
            f"{bx:+.2f}",
            f"{by:+.2f}",
            f"{smaj:.2f}",
            f"{smin:.2f}",
            f"{sphi:.0f}°",
            f"{thetaE:.2f}",
            f"{lx:+.2f}",
            f"{ly:+.2f}",
            f"{q_eff:.2f}",
            f"{lphi_eff:.0f}°",
            f"{noise_sigma:.3f}",
            f"λ={lambda_reg:.2g}",
        )

    return app


def main():
    app = make_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
