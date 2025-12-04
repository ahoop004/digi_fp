from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

try:
    import scipy.ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

# Allow running as a script: python synthetic_lensing/synthetic_gui.py
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from synthetic_lensing.forward import (
    BlurTransform,
    ForwardModel,
    LightProfileSourceModel,
    MaskTransform,
    NoiseTransform,
    PoissonNoiseTransform,
    get_regularizer,
)
from synthetic_lensing.grid import Grid2D
from synthetic_lensing.inverse_api import NaiveInverse, TikhonovInverse
from synthetic_lensing.lens_models import BaseLensModel, SIELens, SISLens
from synthetic_lensing.light_sources import EllipticalGaussianLight, SersicLight


N_PIX_DEFAULT = 48
FOV_DEFAULT = 4.0  # base FOV for pixel scaling
RNG = np.random.default_rng()


def heatmap(arr: np.ndarray, title: str, grid: Grid2D, showscale: bool = False, zmin: float | None = None, zmax: float | None = None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=grid.x,
            y=grid.y,
            colorscale="Viridis",
            showscale=showscale,
            zmin=zmin,
            zmax=zmax,
            zsmooth=False,
        )
    )
    xmin, xmax, ymin, ymax = grid.extent
    pad_x = 0.0
    pad_y = 0.0
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=24, b=24),
        xaxis_title="x",
        yaxis_title="y",
        height=360,
        width=360,
        autosize=False,
    )
    fig.update_xaxes(range=[xmin - pad_x, xmax + pad_x], constrain="domain")
    fig.update_yaxes(range=[ymin - pad_y, ymax + pad_y], scaleanchor="x", scaleratio=1, autorange="reversed")
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


def collapsible_panel(title: str, body: html.Div, open_state: bool = True) -> html.Details:
    return html.Details(
        [
            html.Summary(title, style={"fontWeight": "bold", "cursor": "pointer"}),
            body,
        ],
        open=open_state,
        style={"border": "1px solid #ddd", "borderRadius": "8px", "padding": "6px"},
    )


def build_pipeline(
    n_pix: int,
    fov: float,
    source_profile,
    lens: BaseLensModel,
    lens_light_profile,
    noise_sigma: float,
    use_poisson: bool,
    blur_sigma: float,
    filter_model: str,
    filter_param: float,
    use_block_mask: bool,
    block_radius: float,
    block_center_x: float,
    block_center_y: float,
    block_q: float,
    block_theta_deg: float,
    use_outer_mask: bool,
    outer_radius: float,
    outer_center_x: float,
    outer_center_y: float,
    outer_q: float,
    outer_theta_deg: float,
    lambda_reg: float,
    freeze_noise: bool,
    compute_naive: bool,
    compute_tik: bool,
):
    source_model = LightProfileSourceModel(source_profile)
    transforms = []
    if noise_sigma > 0 and not use_poisson:
        transforms.append(NoiseTransform(noise_sigma, rng=RNG if freeze_noise else None))
    if use_poisson:
        transforms.append(PoissonNoiseTransform(rng=RNG if freeze_noise else None))
    if blur_sigma > 0:
        pix_scale = fov / float(n_pix)
        sigma_px = blur_sigma / pix_scale if pix_scale > 0 else blur_sigma
        transforms.append(BlurTransform(sigma_px))
    fwd = ForwardModel(n_pix=n_pix, fov=fov, source_model=source_model, lens_model=lens, transforms=None)
    sim = fwd.simulate()
    base_image = sim["image_clean"]
    lens_light = lens_light_profile.evaluate(sim["grid"].X, sim["grid"].Y) if lens_light_profile is not None else 0.0
    image_clean = base_image + lens_light
    image_noisy = image_clean
    for t in transforms:
        image_noisy = t.apply(image_noisy)

    image_filtered = image_noisy
    if filter_model == "median" and ndi is not None and filter_param > 0:
        size_px = max(1, int(round(filter_param)))
        image_filtered = ndi.median_filter(image_noisy, size=size_px)
    elif filter_model == "gaussian" and ndi is not None and filter_param > 0:
        image_filtered = ndi.gaussian_filter(image_noisy, sigma=filter_param)

    mask = np.ones_like(image_filtered)
    grid = sim["grid"]
    q_block = np.clip(block_q, 0.2, 1.0)
    q_outer = np.clip(outer_q, 0.2, 1.0)

    def elliptical_r(dx: np.ndarray, dy: np.ndarray, q: float, theta_deg: float) -> np.ndarray:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_p = cos_t * dx + sin_t * dy
        y_p = -sin_t * dx + cos_t * dy
        return np.sqrt(x_p**2 + (y_p / q) ** 2)

    if use_block_mask and block_radius > 0:
        dx = grid.X - block_center_x
        dy = grid.Y - block_center_y
        r_block = elliptical_r(dx, dy, q_block, block_theta_deg)
        mask = np.where(r_block <= block_radius, 0.0, mask)

    if use_outer_mask and outer_radius > 0:
        dx_o = grid.X - outer_center_x
        dy_o = grid.Y - outer_center_y
        r_outer = elliptical_r(dx_o, dy_o, q_outer, outer_theta_deg)
        mask = np.where(r_outer >= outer_radius, 0.0, mask)
    image_obs = image_filtered * mask

    sim["image_clean"] = image_clean
    sim["image_noisy"] = image_noisy
    sim["image_obs"] = image_obs

    source_naive = None
    image_model_naive = None
    if compute_naive:
        naive = NaiveInverse()
        source_naive, image_model_naive = naive.reconstruct(sim["image_obs"], sim["operator"])

    source_tik = None
    image_model_tik = None
    if compute_tik:
        R_reg = get_regularizer(n_pix)
        tik = TikhonovInverse(lambda_reg=lambda_reg)
        source_tik, image_model_tik = tik.reconstruct(sim["image_obs"], sim["operator"], regularizer=R_reg)

    return {
        "source_true": sim["source_true"],
        "image_clean": sim["image_clean"],
        "image_noisy": image_noisy,
        "source_naive": source_naive,
        "source_tik": source_tik,
        "image_model_naive": image_model_naive,
        "image_model_tik": image_model_tik,
        "image_filtered": image_filtered,
        "image_obs": image_obs,
        "grid": sim["grid"],
        "operator": sim["operator"],
        "lens_light": lens_light,
    }


def make_app() -> Dash:
    app = Dash(__name__)

    source_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Source model", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="source-type",
                        options=[
                            {"label": "Elliptical Gaussian", "value": "gaussian"},
                            {"label": "Sersic", "value": "sersic"},
                        ],
                        value="gaussian",
                        clearable=False,
                        style={"width": "180px"},
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
                    html.Button("Centered source", id="preset-src-center", n_clicks=0, className="preset-btn"),
                    html.Button("Offset source", id="preset-src-offset", n_clicks=0, className="preset-btn"),
                ],
                style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginBottom": "8px"},
            ),
            slider_row("i0", "I0", 0.1, 2.0, 0.01, 1.0),
            slider_row("bx", "βx", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("by", "βy", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("sphi", "φ_S", -90, 90, 1, 20, marks={-90: "-90", 0: "0", 90: "90"}),
            html.Div(
                [
                    slider_row("smaj", "σ_maj", 0.05, 1.0, 0.01, 0.4),
                    slider_row("smin", "σ_min", 0.05, 1.0, 0.01, 0.2),
                ],
                id="gaussian-block",
                style={"borderTop": "1px solid #eee", "paddingTop": "6px"},
            ),
            html.Div(
                [
                    slider_row("sersic_reff", "r_eff", 0.05, 1.5, 0.01, 0.5),
                    slider_row("sersic_n", "n", 0.5, 6.0, 0.1, 2.0),
                    slider_row("sersic_q", "q_s", 0.3, 1.0, 0.01, 0.8),
                ],
                id="sersic-block",
                style={"borderTop": "1px solid #eee", "paddingTop": "6px", "display": "none"},
            ),
        ],
        style={"padding": "6px"},
    )
    source_panel = collapsible_panel("Source parameters", source_body, open_state=False)

    lens_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Lens model", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="lens-type",
                        options=[
                            {"label": "SIS", "value": "sis"},
                            {"label": "SIE", "value": "sie"},
                        ],
                        value="sis",
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
            slider_row("thetaE", "θ_E", 0.5, 2.5, 0.01, 1.2),
            slider_row("lx", "x0", -0.5, 0.5, 0.01, 0.0, marks=simple_marks(-0.5, 0.5)),
            slider_row("ly", "y0", -0.5, 0.5, 0.01, 0.0, marks=simple_marks(-0.5, 0.5)),
            slider_row("q", "q", 0.5, 1.0, 0.01, 0.85, marks={0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}),
            slider_row("lphi", "φ_L", -45, 45, 1, 10, marks={-45: "-45", 0: "0", 45: "45"}),
        ],
        style={"padding": "6px"},
    )
    lens_panel = collapsible_panel("Lens parameters", lens_body, open_state=False)

    lens_light_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Model", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="ll-type",
                        options=[
                            {"label": "Elliptical Gaussian", "value": "gaussian"},
                            {"label": "Sersic", "value": "sersic"},
                        ],
                        value="gaussian",
                        clearable=False,
                        style={"width": "180px"},
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
            slider_row("ll_i0", "I0_L", 0.0, 2.0, 0.01, 0.2),
            slider_row("ll_dx", "Δx_L", -0.5, 0.5, 0.01, 0.0),
            slider_row("ll_dy", "Δy_L", -0.5, 0.5, 0.01, 0.0),
            dcc.Checklist(
                id="ll-lock-center",
                options=[{"label": "Lock to lens center", "value": "lock"}],
                value=[],
                style={"fontSize": "11px", "marginLeft": "4px"},
            ),
            html.Div(
                [
                    slider_row("ll_smaj", "σ_L,maj", 0.05, 1.0, 0.01, 0.2),
                    slider_row("ll_smin", "σ_L,min", 0.05, 1.0, 0.01, 0.2),
                    slider_row("ll_phi", "φ_L,light", -90, 90, 1, 0.0, marks={-90: "-90", 0: "0", 90: "90"}),
                ],
                id="ll-gaussian-block",
                style={"borderTop": "1px solid #eee", "paddingTop": "6px"},
            ),
            html.Div(
                [
                    slider_row("ll_reff", "r_eff,L", 0.05, 1.5, 0.01, 0.5),
                    slider_row("ll_n", "n_L", 0.5, 6.0, 0.1, 2.0),
                    slider_row("ll_q", "q_L", 0.3, 1.0, 0.01, 0.8),
                ],
                id="ll-sersic-block",
                style={"borderTop": "1px solid #eee", "paddingTop": "6px", "display": "none"},
            ),
        ],
        style={"padding": "6px"},
    )
    lens_light_panel = collapsible_panel("Lens light", lens_light_body, open_state=False)

    noise_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Noise model", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="noise-model",
                        options=[
                            {"label": "Gaussian", "value": "gaussian"},
                            {"label": "Poisson", "value": "poisson"},
                        ],
                        value="gaussian",
                        clearable=False,
                        style={"width": "140px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "90px 1fr",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginTop": "8px",
                },
            ),
            slider_row("noise", "σ_noise", 0.0, 0.1, 0.001, 0.02, marks={0.0: "0", 0.05: "0.05", 0.1: "0.1"}),
            html.Div(
                [
                    html.Div("Blur", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="blur-model",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Gaussian blur", "value": "gaussian"},
                        ],
                        value="none",
                        clearable=False,
                        style={"width": "140px"},
                    )
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "90px 1fr",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginTop": "8px",
                },
            ),
            slider_row("blur_sigma", "σ_blur", 0.0, 1.0, 0.01, 0.0),
        ],
        style={"padding": "6px"},
    )
    noise_panel = collapsible_panel("Noise / Blur", noise_body, open_state=False)

    filter_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Filter", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="filter-model",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Median", "value": "median"},
                            {"label": "Gaussian", "value": "gaussian"},
                        ],
                        value="none",
                        clearable=False,
                        style={"width": "140px"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "90px 1fr",
                    "alignItems": "center",
                    "gap": "6px",
                    "marginTop": "8px",
                },
            ),
            slider_row("filter-param", "Filter param", 0.0, 5.0, 0.1, 0.0),
        ],
        style={"padding": "6px"},
    )
    filter_panel = collapsible_panel("Filters", filter_body, open_state=False)

    mask_center_body = html.Div(
        [
            dcc.Checklist(
                id="mask-center-toggle",
                options=[{"label": "Enable center mask", "value": "enable"}],
                value=[],
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),
            slider_row("mask_block_radius", "Block r", 0.0, 1.5, 0.01, 0.2),
            slider_row("mask_center_x", "x0", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("mask_center_y", "y0", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("mask_q", "q", 0.2, 1.0, 0.01, 1.0, marks={0.2: "0.2", 0.6: "0.6", 1.0: "1.0"}),
            slider_row("mask_theta", "θ (deg)", -90, 90, 1, 0.0, marks={-90: "-90", 0: "0", 90: "90"}),
        ],
        style={"padding": "6px"},
    )
    mask_center_panel = collapsible_panel("Center mask", mask_center_body, open_state=False)

    mask_outer_body = html.Div(
        [
            dcc.Checklist(
                id="mask-outer-toggle",
                options=[{"label": "Enable outer mask", "value": "enable"}],
                value=[],
                style={"fontSize": "11px", "marginBottom": "8px"},
            ),
            slider_row("mask_outer_radius", "Outer r", 0.5, 3.0, 0.01, 2.0),
            slider_row("mask_outer_x", "x0", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("mask_outer_y", "y0", -1.5, 1.5, 0.01, 0.0, marks=simple_marks(-1.5, 1.5)),
            slider_row("mask_outer_q", "q", 0.2, 1.0, 0.01, 1.0, marks={0.2: "0.2", 0.6: "0.6", 1.0: "1.0"}),
            slider_row("mask_outer_theta", "θ (deg)", -90, 90, 1, 0.0, marks={-90: "-90", 0: "0", 90: "90"}),
        ],
        style={"padding": "6px"},
    )
    mask_outer_panel = collapsible_panel("Outer mask", mask_outer_body, open_state=False)

    recon_body = html.Div(
        [
            html.Div(
                [
                    html.Div("Method", style={"width": "90px"}),
                    dcc.Dropdown(
                        id="recon-method",
                        options=[
                            {"label": "Naive", "value": "naive"},
                            {"label": "Tikhonov", "value": "tik"},
                        ],
                        value="tik",
                        clearable=False,
                        style={"width": "160px"},
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
            slider_row(
                "lambda_log",
                "log10 λ",
                -4,
                0,
                0.1,
                -2,
                marks={-4: "1e-4", -3: "1e-3", -2: "1e-2", -1: "1e-1", 0: "1"},
            ),
        ],
        style={"padding": "6px"},
    )
    recon_panel = collapsible_panel("Reconstruction", recon_body, open_state=False)

    grid_body = html.Div(
        [
            slider_row("resolution", "Resolution (px)", 32, 128, 1, N_PIX_DEFAULT),
            slider_row("fov", "FOV", 2.0, 8.0, 0.1, FOV_DEFAULT),
        ],
        style={"padding": "0px"},
    )
    grid_panel = collapsible_panel("Grid / Scale", grid_body, open_state=False)

    metrics_body = html.Div(
        [
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
                        style={"fontSize": "11px"},
                    )
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(id="metrics-text", style={"fontSize": "11px"}),
        ],
        style={"padding": "0px"},
    )
    metrics_panel = collapsible_panel("Metrics / Residual", metrics_body, open_state=False)

    plots = html.Div(
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
                    dcc.Graph(id="filtered-image"),
                    dcc.Graph(id="masked-image"),
                    dcc.Graph(id="recon-figure"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "12px",
                    "marginTop": "12px",
                },
            ),
        ]
    )

    controls_column1 = html.Div(
        [source_panel, lens_panel, lens_light_panel, grid_panel], style={"display": "grid", "gap": "12px", "fontSize": "12px"}
    )
    controls_column2 = html.Div(
        [noise_panel, filter_panel, mask_center_panel, mask_outer_panel, recon_panel, metrics_panel],
        style={"display": "grid", "gap": "12px", "fontSize": "12px"},
    )

    app.layout = html.Div(
        [
            html.H2("Gravitational Lensing"),
            html.Div(
                [controls_column1, controls_column2, plots],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "340px 340px 1fr",
                    "gap": "12px",
                    "alignItems": "start",
                },
            ),
        ],
        style={"maxWidth": "min(1600px, 100vw - 24px)", "margin": "0 auto", "padding": "12px"},
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
        State("bx", "value"),
        State("by", "value"),
        State("sphi", "value"),
        State("lens-type", "value"),
        State("q", "value"),
        State("lphi", "value"),
    )
    def apply_presets(n_center, n_offset, bx, by, sphi, lens_type, q, lphi):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trig == "preset-src-center":
            bx, by, sphi = 0.0, 0.0, 0.0
        elif trig == "preset-src-offset":
            bx, by, sphi = 0.6, -0.3, 20.0
        return bx, by, sphi, lens_type, q, lphi

    @app.callback(
        Output("gaussian-block", "style"),
        Output("sersic-block", "style"),
        Input("source-type", "value"),
    )
    def toggle_source_blocks(source_type):
        base_style = {"borderTop": "1px solid #eee", "paddingTop": "6px"}
        if source_type == "sersic":
            return {**base_style, "display": "none"}, {**base_style, "display": "block"}
        return {**base_style, "display": "block"}, {**base_style, "display": "none"}

    @app.callback(
        Output("ll-gaussian-block", "style"),
        Output("ll-sersic-block", "style"),
        Input("ll-type", "value"),
    )
    def toggle_ll_blocks(ll_type):
        base_style = {"borderTop": "1px solid #eee", "paddingTop": "6px"}
        if ll_type == "sersic":
            return {**base_style, "display": "none"}, {**base_style, "display": "block"}
        return {**base_style, "display": "block"}, {**base_style, "display": "none"}

    @app.callback(
        Output("true-source", "figure"),
        Output("lensed-clean", "figure"),
        Output("lensed-noisy", "figure"),
        Output("filtered-image", "figure"),
        Output("masked-image", "figure"),
        Output("recon-figure", "figure"),
        Output("i0-val", "children"),
        Output("bx-val", "children"),
        Output("by-val", "children"),
        Output("smaj-val", "children"),
        Output("smin-val", "children"),
        Output("sphi-val", "children"),
        Output("sersic_reff-val", "children"),
        Output("sersic_n-val", "children"),
        Output("sersic_q-val", "children"),
        Output("ll_i0-val", "children"),
        Output("ll_dx-val", "children"),
        Output("ll_dy-val", "children"),
        Output("ll_dx", "disabled"),
        Output("ll_dy", "disabled"),
        Output("ll_smaj-val", "children"),
        Output("ll_smin-val", "children"),
        Output("ll_phi-val", "children"),
        Output("thetaE-val", "children"),
        Output("lx-val", "children"),
        Output("ly-val", "children"),
        Output("q-val", "children"),
        Output("lphi-val", "children"),
        Output("noise-val", "children"),
        Output("mask_block_radius-val", "children"),
        Output("mask_center_x-val", "children"),
        Output("mask_center_y-val", "children"),
        Output("mask_q-val", "children"),
        Output("mask_theta-val", "children"),
        Output("mask_outer_radius-val", "children"),
        Output("mask_outer_x-val", "children"),
        Output("mask_outer_y-val", "children"),
        Output("mask_outer_q-val", "children"),
        Output("mask_outer_theta-val", "children"),
        Output("lambda_log-val", "children"),
        Output("metrics-text", "children"),
        Input("source-type", "value"),
        Input("ll-type", "value"),
        Input("lens-type", "value"),
        Input("resolution", "value"),
        Input("fov", "value"),
        Input("residual-mode", "value"),
        Input("recon-method", "value"),
        Input("noise-model", "value"),
        Input("blur-model", "value"),
        Input("filter-model", "value"),
        Input("filter-param", "value"),
        Input("mask-center-toggle", "value"),
        Input("mask-outer-toggle", "value"),
        Input("mask_center_x", "value"),
        Input("mask_center_y", "value"),
        Input("mask_q", "value"),
        Input("mask_theta", "value"),
        Input("mask_outer_x", "value"),
        Input("mask_outer_y", "value"),
        Input("mask_outer_q", "value"),
        Input("mask_outer_theta", "value"),
        Input("preset-src-center", "n_clicks"),
        Input("preset-src-offset", "n_clicks"),
        Input("i0", "value"),
        Input("bx", "value"),
        Input("by", "value"),
        Input("smaj", "value"),
        Input("smin", "value"),
        Input("sphi", "value"),
        Input("sersic_reff", "value"),
        Input("sersic_n", "value"),
        Input("sersic_q", "value"),
        Input("ll_i0", "value"),
        Input("ll_dx", "value"),
        Input("ll_dy", "value"),
        Input("ll-lock-center", "value"),
        Input("ll_smaj", "value"),
        Input("ll_smin", "value"),
        Input("ll_phi", "value"),
        Input("ll_reff", "value"),
        Input("ll_n", "value"),
        Input("ll_q", "value"),
        Input("thetaE", "value"),
        Input("lx", "value"),
        Input("ly", "value"),
        Input("q", "value"),
        Input("lphi", "value"),
        Input("noise", "value"),
        Input("blur_sigma", "value"),
        Input("mask_block_radius", "value"),
        Input("mask_outer_radius", "value"),
        Input("lambda_log", "value"),
    )
    def update_figs(
        source_type,
        ll_type,
        lens_type,
        n_pix,
        fov,
        residual_mode,
        recon_method,
        noise_model,
        blur_model,
        filter_model,
        filter_param,
        mask_center_toggle,
        mask_outer_toggle,
        mask_center_x,
        mask_center_y,
        mask_q,
        mask_theta,
        mask_outer_x,
        mask_outer_y,
        mask_outer_q,
        mask_outer_theta,
        n_center,
        n_offset,
        i0,
        bx,
        by,
        smaj,
        smin,
        sphi,
        sersic_reff,
        sersic_n,
        sersic_q,
        ll_i0,
        ll_dx,
        ll_dy,
        ll_lock_center,
        ll_smaj,
        ll_smin,
        ll_phi,
        ll_reff,
        ll_n,
        ll_q,
        thetaE,
        lx,
        ly,
        q,
        lphi,
        noise_sigma,
        blur_sigma,
        mask_block_radius,
        mask_outer_radius,
        lambda_log,
    ):
        # Normalize inputs
        recon_method = recon_method or "tik"
        base_fov = FOV_DEFAULT
        fov = float(fov)
        # Keep pixel scale roughly constant as FOV changes
        n_pix = int(max(16, round(float(n_pix) * (fov / base_fov))))
        mask_center_x = float(mask_center_x or 0.0)
        mask_center_y = float(mask_center_y or 0.0)
        mask_q = float(mask_q or 1.0)
        mask_theta = float(mask_theta or 0.0)
        mask_outer_x = float(mask_outer_x or 0.0)
        mask_outer_y = float(mask_outer_y or 0.0)
        mask_outer_q = float(mask_outer_q or 1.0)
        mask_outer_theta = float(mask_outer_theta or 0.0)

        # Apply preset only if a preset button triggered this callback
        trig_id = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
        if trig_id in {"preset-src-center", "preset-src-offset"}:
            if trig_id == "preset-src-center":
                bx, by, sphi = 0.0, 0.0, 0.0
            elif trig_id == "preset-src-offset":
                bx, by, sphi = 0.6, -0.3, 20.0

        if source_type == "sersic":
            source_profile = SersicLight(
                I0=i0,
                x0=bx,
                y0=by,
                r_eff=sersic_reff,
                n=sersic_n,
                q=sersic_q,
                phi_deg=sphi,
            )
        else:
            source_profile = EllipticalGaussianLight(
                I0=i0, x0=bx, y0=by, sigma_major=smaj, sigma_minor=smin, phi_deg=sphi
            )
        if lens_type == "sis":
            lens = SISLens(theta_E=thetaE, x0=lx, y0=ly)
            q_eff, lphi_eff = 1.0, 0.0
        else:
            lens = SIELens(theta_E=thetaE, x0=lx, y0=ly, q=q, phi_deg=lphi)
            q_eff, lphi_eff = q, lphi

        lock_ll = "lock" in (ll_lock_center or [])
        ll_dx_eff = 0.0 if lock_ll else ll_dx
        ll_dy_eff = 0.0 if lock_ll else ll_dy

        if ll_type == "sersic":
            lens_light_profile = SersicLight(
                I0=ll_i0,
                x0=lx + ll_dx_eff,
                y0=ly + ll_dy_eff,
                r_eff=ll_reff,
                n=ll_n,
                q=ll_q,
                phi_deg=ll_phi,
            )
        else:
            lens_light_profile = EllipticalGaussianLight(
                I0=ll_i0,
                x0=lx + ll_dx_eff,
                y0=ly + ll_dy_eff,
                sigma_major=ll_smaj,
                sigma_minor=ll_smin,
                phi_deg=ll_phi,
            )
        lambda_reg = 10 ** lambda_log
        use_poisson = noise_model == "poisson"
        freeze_noise = False  # freeze option removed
        blur_sigma_eff = blur_sigma if blur_model == "gaussian" else 0.0
        use_block_mask = "enable" in (mask_center_toggle or [])
        use_outer_mask = "enable" in (mask_outer_toggle or [])
        compute_naive = recon_method == "naive"
        compute_tik = recon_method == "tik"
        filter_model = filter_model or "none"
        filter_param = float(filter_param or 0.0)

        data = build_pipeline(
            n_pix,
            fov,
            source_profile,
            lens,
            lens_light_profile,
            noise_sigma,
            use_poisson,
            blur_sigma_eff,
            filter_model,
            filter_param,
            use_block_mask,
            mask_block_radius,
            mask_center_x,
            mask_center_y,
            mask_q,
            mask_theta,
            use_outer_mask,
            mask_outer_radius,
            mask_outer_x,
            mask_outer_y,
            mask_outer_q,
            mask_outer_theta,
            lambda_reg,
            freeze_noise,
            compute_naive,
            compute_tik,
        )

        grid = data["grid"]

        metrics = []
        if data["source_naive"] is not None:
            metrics.append(f"MSE_naive={float(np.mean((data['source_true'] - data['source_naive'])**2)):.3e}")
        if data["source_tik"] is not None:
            metrics.append(f"MSE_tik={float(np.mean((data['source_true'] - data['source_tik'])**2)):.3e}")

        recon_fig = None
        if recon_method == "naive" and data["source_naive"] is not None:
            recon_fig = heatmap(data["source_naive"], "Reconstruction (Naive)", grid, zmin=0.0, zmax=2.0)
        elif recon_method == "tik" and data["source_tik"] is not None:
            recon_fig = heatmap(data["source_tik"], "Reconstruction (Tikhonov)", grid, zmin=0.0, zmax=2.0)
        else:
            recon_fig = empty_heatmap("Reconstruction", grid)

        return (
            heatmap(data["source_true"], "True source", grid, zmin=0.0, zmax=2.0),
            heatmap(data["image_clean"], "Lensed clean", grid, zmin=0.0, zmax=2.0),
            heatmap(data["image_noisy"], "Lensed noisy", grid, zmin=0.0, zmax=2.0),
            heatmap(data["image_filtered"], "Filtered (post noise/blur)", grid, zmin=0.0, zmax=2.0),
            heatmap(data["image_obs"], "Masked / observed", grid, zmin=0.0, zmax=2.0),
            recon_fig,
            f"{i0:.2f}",
            f"{bx:+.2f}",
            f"{by:+.2f}",
            f"{smaj:.2f}",
            f"{smin:.2f}",
            f"{sphi:.0f}°",
            f"{sersic_reff:.2f}",
            f"{sersic_n:.2f}",
            f"{sersic_q:.2f}",
            f"{ll_i0:.2f}",
            f"{ll_dx_eff:+.2f}",
            f"{ll_dy_eff:+.2f}",
            lock_ll,
            lock_ll,
            f"{ll_smaj:.2f}",
            f"{ll_smin:.2f}",
            f"{ll_phi:.0f}°",
            f"{thetaE:.2f}",
            f"{lx:+.2f}",
            f"{ly:+.2f}",
            f"{q_eff:.2f}",
            f"{lphi_eff:.0f}°",
            f"{noise_sigma:.3f}",
            f"{mask_block_radius:.2f}",
            f"{mask_center_x:+.2f}",
            f"{mask_center_y:+.2f}",
            f"{mask_q:.2f}",
            f"{mask_theta:.0f}°",
            f"{mask_outer_radius:.2f}",
            f"{mask_outer_x:+.2f}",
            f"{mask_outer_y:+.2f}",
            f"{mask_outer_q:.2f}",
            f"{mask_outer_theta:.0f}°",
            f"λ={lambda_reg:.2g}",
            ", ".join(metrics) if metrics else "",
        )

    return app


def main():
    app = make_app()
    app.run(debug=True)


if __name__ == "__main__":
    main()
