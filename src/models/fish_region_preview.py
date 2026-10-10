from __future__ import annotations

"""
Real-scale 2D fish mucus penetration ABM.

Outputs only three scientific figures:
1) whole-fish geometry and 600-um mucus definition;
2) whole-fish bacterial occupancy heatmap;
3) separate head / side / rear 600-um penetration heatmaps.

Default geometry:
- fish length: 100 mm
- fish height: 30 mm
- mucus thickness: 600 um = 0.6 mm
- head is on the left, rear is on the right
- relative water flow is along +x by default
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
import numpy as np


# Editable SVG text and Chinese-capable sans-serif stack.
_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_CJK_FONT_PATH)
_CJK_FONT_NAME = font_manager.FontProperties(fname=_CJK_FONT_PATH).get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    _CJK_FONT_NAME,
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


PALETTE = {
    "fish": "#3A8D8A",
    "fish_edge": "#245C5A",
    "mucus": "#DCE6F7",
    "mucus_edge": "#5969B3",
    "head": "#0F4D92",
    "side": "#C97B2A",
    "rear": "#8057A6",
    "accent": "#B64342",
    "neutral": "#606060",
}


@dataclass
class Params:
    OUTPUT_DIR: str = str(Path(__file__).resolve().parents[2] / "results" / "fish_region")
    SEED: int = 13

    # Realistic macroscopic body scale.
    FISH_LENGTH_MM: float = 100.0
    FISH_HEIGHT_MM: float = 30.0
    MUCUS_THICKNESS_UM: float = 600.0

    # Same global flow is applied to head, side and rear runs.
    FLOW_ANGLE_DEG: float = 0.0
    FLOW_SPEED_UM_S: float = 12.0

    # Agent-based motility.
    N_PER_REGION: int = 3200
    T_S: float = 70.0
    DT_S: float = 0.05
    SWIM_SPEED_WATER_UM_S: float = 34.0
    SWIM_SPEED_MUCUS_OUTER_UM_S: float = 29.0
    SWIM_SPEED_MUCUS_INNER_UM_S: float = 18.0
    ROT_DIFF_S: float = 0.10
    TUMBLE_RATE_S: float = 0.28
    INITIAL_INWARD_KAPPA: float = 7.0
    TUMBLE_INWARD_KAPPA: float = 1.4

    # Reduced-order near-body flow.
    MUCUS_TANGENTIAL_FLOW_FLOOR: float = 0.28
    MUCUS_NORMAL_FLOW_FLOOR: float = 0.08
    NORMAL_SUPPRESSION_SCALE_UM: float = 90.0

    # Region sampling around ellipse parameter angles.
    HEAD_HALF_ANGLE_DEG: float = 14.0
    SIDE_HALF_ANGLE_DEG: float = 7.0
    REAR_HALF_ANGLE_DEG: float = 14.0

    # Local heatmap frame.
    LOCAL_HALF_WIDTH_MM: float = 7.0
    LOCAL_X_BINS: int = 150
    LOCAL_DEPTH_BINS: int = 120
    FULL_X_BINS: int = 520
    FULL_Y_BINS: int = 220
    HIST_STRIDE: int = 4

    @property
    def a_um(self) -> float:
        return 0.5 * self.FISH_LENGTH_MM * 1000.0

    @property
    def b_um(self) -> float:
        return 0.5 * self.FISH_HEIGHT_MM * 1000.0


HP = Params()


REGIONS = {
    # Head is left, side is dorsal/top, rear is right.
    "head": {"label": "头部", "t0": np.pi, "half_angle_key": "HEAD_HALF_ANGLE_DEG", "color": PALETTE["head"]},
    "side": {"label": "侧面", "t0": 0.5 * np.pi, "half_angle_key": "SIDE_HALF_ANGLE_DEG", "color": PALETTE["side"]},
    "rear": {"label": "后部", "t0": 0.0, "half_angle_key": "REAR_HALF_ANGLE_DEG", "color": PALETTE["rear"]},
}


def ellipse_point_normal_tangent(t: np.ndarray | float, hp: Params):
    t = np.asarray(t, dtype=float)
    x = hp.a_um * np.cos(t)
    y = hp.b_um * np.sin(t)

    gx = np.cos(t) / hp.a_um
    gy = np.sin(t) / hp.b_um
    gn = np.sqrt(gx * gx + gy * gy)
    nx = gx / gn
    ny = gy / gn
    tx = -ny
    ty = nx
    return x, y, nx, ny, tx, ty


def approximate_signed_distance(x: np.ndarray, y: np.ndarray, hp: Params) -> np.ndarray:
    """First-order ellipse signed distance, accurate in the 600-um near-surface shell."""
    phi = (x / hp.a_um) ** 2 + (y / hp.b_um) ** 2 - 1.0
    gx = 2.0 * x / hp.a_um**2
    gy = 2.0 * y / hp.b_um**2
    return phi / (np.sqrt(gx * gx + gy * gy) + 1e-12)


def local_normal_tangent(x: np.ndarray, y: np.ndarray, hp: Params):
    gx = 2.0 * x / hp.a_um**2
    gy = 2.0 * y / hp.b_um**2
    gn = np.sqrt(gx * gx + gy * gy) + 1e-12
    nx = gx / gn
    ny = gy / gn
    tx = -ny
    ty = nx
    return nx, ny, tx, ty


def outer_offset_curve(hp: Params, n: int = 1800):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y, nx, ny, _, _ = ellipse_point_normal_tangent(t, hp)
    return x + hp.MUCUS_THICKNESS_UM * nx, y + hp.MUCUS_THICKNESS_UM * ny


def inner_curve(hp: Params, n: int = 1800):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y, _, _, _, _ = ellipse_point_normal_tangent(t, hp)
    return x, y


def flow_field(x: np.ndarray, y: np.ndarray, hp: Params):
    angle = np.deg2rad(hp.FLOW_ANGLE_DEG)
    ux_inf = hp.FLOW_SPEED_UM_S * np.cos(angle)
    uy_inf = hp.FLOW_SPEED_UM_S * np.sin(angle)

    d = approximate_signed_distance(x, y, hp)
    nx, ny, tx, ty = local_normal_tangent(x, y, hp)
    un = ux_inf * nx + uy_inf * ny
    ut = ux_inf * tx + uy_inf * ty

    ux = np.full_like(x, ux_inf, dtype=float)
    uy = np.full_like(y, uy_inf, dtype=float)

    in_mucus = (d > 0.0) & (d <= hp.MUCUS_THICKNESS_UM)
    if np.any(in_mucus):
        r = np.clip(d[in_mucus] / hp.MUCUS_THICKNESS_UM, 0.0, 1.0)
        tangential_factor = hp.MUCUS_TANGENTIAL_FLOW_FLOOR + (1.0 - hp.MUCUS_TANGENTIAL_FLOW_FLOOR) * r
        normal_wall = np.tanh(d[in_mucus] / hp.NORMAL_SUPPRESSION_SCALE_UM)
        normal_factor = (
            hp.MUCUS_NORMAL_FLOW_FLOOR
            + (1.0 - hp.MUCUS_NORMAL_FLOW_FLOOR) * r
        ) * normal_wall

        ux[in_mucus] = (
            normal_factor * un[in_mucus] * nx[in_mucus]
            + tangential_factor * ut[in_mucus] * tx[in_mucus]
        )
        uy[in_mucus] = (
            normal_factor * un[in_mucus] * ny[in_mucus]
            + tangential_factor * ut[in_mucus] * ty[in_mucus]
        )

    in_tissue = d <= 0.0
    ux[in_tissue] = 0.0
    uy[in_tissue] = 0.0
    return ux, uy, d


def swim_speed(d: np.ndarray, hp: Params) -> np.ndarray:
    v = np.full_like(d, hp.SWIM_SPEED_WATER_UM_S, dtype=float)
    in_mucus = (d > 0.0) & (d <= hp.MUCUS_THICKNESS_UM)
    if np.any(in_mucus):
        r = np.clip(d[in_mucus] / hp.MUCUS_THICKNESS_UM, 0.0, 1.0)
        v[in_mucus] = (
            hp.SWIM_SPEED_MUCUS_INNER_UM_S
            + (hp.SWIM_SPEED_MUCUS_OUTER_UM_S - hp.SWIM_SPEED_MUCUS_INNER_UM_S) * r
        )
    v[d <= 0.0] = 0.0
    return v


def initialize_region(region_name: str, hp: Params, rng: np.random.Generator):
    spec = REGIONS[region_name]
    t0 = float(spec["t0"])
    half_angle = np.deg2rad(float(getattr(hp, spec["half_angle_key"])))
    t = rng.uniform(t0 - half_angle, t0 + half_angle, size=hp.N_PER_REGION)

    bx, by, nx, ny, _, _ = ellipse_point_normal_tangent(t, hp)
    outside_gap = hp.MUCUS_THICKNESS_UM + rng.uniform(35.0, 170.0, size=hp.N_PER_REGION)
    x = bx + outside_gap * nx
    y = by + outside_gap * ny

    inward_angle = np.arctan2(-ny, -nx)
    theta = rng.vonmises(inward_angle, hp.INITIAL_INWARD_KAPPA)
    return x, y, theta


def region_reference(region_name: str, hp: Params):
    t0 = float(REGIONS[region_name]["t0"])
    x0, y0, nx0, ny0, tx0, ty0 = ellipse_point_normal_tangent(t0, hp)
    return np.array([float(x0), float(y0)]), np.array([float(nx0), float(ny0)]), np.array([float(tx0), float(ty0)])


def to_local_coordinates(x, y, region_name: str, d, hp: Params):
    p0, _, tangent = region_reference(region_name, hp)
    dx = x - p0[0]
    dy = y - p0[1]
    tangent_mm = (dx * tangent[0] + dy * tangent[1]) / 1000.0
    penetration_depth_um = hp.MUCUS_THICKNESS_UM - d
    return tangent_mm, penetration_depth_um


def simulate_region(region_name: str, hp: Params, seed_offset: int = 0) -> Dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(hp.SEED + seed_offset)
    x, y, theta = initialize_region(region_name, hp, rng)
    alive = np.ones(hp.N_PER_REGION, dtype=bool)
    penetrated = np.zeros(hp.N_PER_REGION, dtype=bool)

    local_x_edges = np.linspace(-hp.LOCAL_HALF_WIDTH_MM, hp.LOCAL_HALF_WIDTH_MM, hp.LOCAL_X_BINS + 1)
    depth_edges = np.linspace(0.0, hp.MUCUS_THICKNESS_UM, hp.LOCAL_DEPTH_BINS + 1)
    local_hist = np.zeros((hp.LOCAL_DEPTH_BINS, hp.LOCAL_X_BINS), dtype=float)

    full_x_edges = np.linspace(-0.53 * hp.FISH_LENGTH_MM, 0.53 * hp.FISH_LENGTH_MM, hp.FULL_X_BINS + 1)
    full_y_edges = np.linspace(-0.58 * hp.FISH_HEIGHT_MM, 0.58 * hp.FISH_HEIGHT_MM, hp.FULL_Y_BINS + 1)
    full_hist = np.zeros((hp.FULL_Y_BINS, hp.FULL_X_BINS), dtype=float)

    steps = int(round(hp.T_S / hp.DT_S))
    for k in range(steps):
        active = alive.copy()
        if not np.any(active):
            break

        ux, uy, d_prev = flow_field(x, y, hp)
        nx, ny, _, _ = local_normal_tangent(x, y, hp)

        tumble_prob = 1.0 - np.exp(-hp.TUMBLE_RATE_S * hp.DT_S)
        tumble = active & (rng.random(hp.N_PER_REGION) < tumble_prob)
        if np.any(tumble):
            inward_angle = np.arctan2(-ny[tumble], -nx[tumble])
            theta[tumble] = rng.vonmises(inward_angle, hp.TUMBLE_INWARD_KAPPA)

        theta[active] += np.sqrt(2.0 * hp.ROT_DIFF_S * hp.DT_S) * rng.standard_normal(np.sum(active))
        v = swim_speed(d_prev, hp)
        x[active] += (ux[active] + v[active] * np.cos(theta[active])) * hp.DT_S
        y[active] += (uy[active] + v[active] * np.sin(theta[active])) * hp.DT_S

        d_now = approximate_signed_distance(x, y, hp)

        newly_penetrated = active & (d_prev > 0.0) & (d_now <= 0.0)
        penetrated[newly_penetrated] = True
        alive[newly_penetrated] = False

        # Accumulate heatmaps only in the 600-um mucus layer.
        if k % max(1, hp.HIST_STRIDE) == 0:
            in_mucus = alive & (d_now > 0.0) & (d_now <= hp.MUCUS_THICKNESS_UM)
            if np.any(in_mucus):
                lx, depth = to_local_coordinates(x[in_mucus], y[in_mucus], region_name, d_now[in_mucus], hp)
                h_local, _, _ = np.histogram2d(depth, lx, bins=[depth_edges, local_x_edges])
                local_hist += h_local

                h_full, _, _ = np.histogram2d(
                    y[in_mucus] / 1000.0,
                    x[in_mucus] / 1000.0,
                    bins=[full_y_edges, full_x_edges],
                )
                full_hist += h_full

        lx_all, _ = to_local_coordinates(x, y, region_name, d_now, hp)
        escaped = alive & ((d_now > hp.MUCUS_THICKNESS_UM + 1300.0) | (np.abs(lx_all) > hp.LOCAL_HALF_WIDTH_MM + 2.0))
        alive[escaped] = False

    return {
        "local_hist": local_hist,
        "local_x_edges": local_x_edges,
        "depth_edges": depth_edges,
        "full_hist": full_hist,
        "full_x_edges": full_x_edges,
        "full_y_edges": full_y_edges,
        "penetration_fraction": float(np.mean(penetrated)),
        "penetrated_count": int(np.sum(penetrated)),
        "n": int(hp.N_PER_REGION),
    }


def add_panel_label(ax: plt.Axes, label: str):
    ax.text(-0.08, 1.03, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_figure(fig: plt.Figure, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def assert_axes_do_not_overlap(fig: plt.Figure, figure_name: str):
    fig.canvas.draw()
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    boxes = [ax.get_position() for ax in visible_axes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # Colorbar axes can share a boundary but should not have a positive-area overlap.
            x_overlap = min(boxes[i].x1, boxes[j].x1) - max(boxes[i].x0, boxes[j].x0)
            y_overlap = min(boxes[i].y1, boxes[j].y1) - max(boxes[i].y0, boxes[j].y0)
            if x_overlap > 1e-4 and y_overlap > 1e-4:
                raise RuntimeError(f"Layout overlap detected in {figure_name}: axes {i} and {j}")


def draw_fish(ax: plt.Axes, hp: Params, fill_mucus: bool = True):
    xo, yo = outer_offset_curve(hp)
    xi, yi = inner_curve(hp)
    if fill_mucus:
        ax.fill(xo / 1000.0, yo / 1000.0, color=PALETTE["mucus"], zorder=1)
    ax.fill(xi / 1000.0, yi / 1000.0, color=PALETTE["fish"], zorder=3)
    ax.plot(xo / 1000.0, yo / 1000.0, color=PALETTE["mucus_edge"], lw=1.1, zorder=4)
    ax.plot(xi / 1000.0, yi / 1000.0, color=PALETTE["fish_edge"], lw=1.0, zorder=4)
    ax.set_aspect("equal", adjustable="box")


def region_arc(region_name: str, hp: Params, offset_um: float = 600.0, n: int = 160):
    spec = REGIONS[region_name]
    t0 = float(spec["t0"])
    half_angle = np.deg2rad(float(getattr(hp, spec["half_angle_key"])))
    t = np.linspace(t0 - half_angle, t0 + half_angle, n)
    x, y, nx, ny, _, _ = ellipse_point_normal_tangent(t, hp)
    return (x + offset_um * nx) / 1000.0, (y + offset_um * ny) / 1000.0


def plot_geometry(hp: Params, output_dir: Path):
    fig = plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_scale = fig.add_subplot(gs[0, 1])

    draw_fish(ax, hp, fill_mucus=True)
    for region_name, spec in REGIONS.items():
        xa, ya = region_arc(region_name, hp)
        ax.plot(xa, ya, lw=4.0, color=spec["color"], solid_capstyle="round", zorder=6)

    ax.text(-52.5, -1.0, "头部", color=PALETTE["head"], ha="right", va="center", fontweight="bold")
    ax.text(0.0, 17.2, "侧面", color=PALETTE["side"], ha="center", va="bottom", fontweight="bold")
    ax.text(52.5, -1.0, "后部", color=PALETTE["rear"], ha="left", va="center", fontweight="bold")
    ax.text(0.0, 0.0, "鱼体：100 mm × 30 mm", color="white", ha="center", va="center", fontweight="bold")

    ax.set_xlim(-55.0, 55.0)
    ax.set_ylim(-18.5, 18.5)
    ax.set_xlabel("鱼体纵轴位置（mm）")
    ax.set_ylabel("横向位置（mm）")
    ax.set_title("正常尺度二维鱼体与头部、侧面、后部取样区")
    add_panel_label(ax, "a")

    # Dedicated scale panel: no inset overlap with the whole-fish panel.
    ax_scale.axvspan(-300.0, 0.0, color=PALETTE["fish"], alpha=0.95)
    ax_scale.axvspan(0.0, hp.MUCUS_THICKNESS_UM, color=PALETTE["mucus"], alpha=1.0)
    ax_scale.axvspan(hp.MUCUS_THICKNESS_UM, 900.0, color="#F7FAFC", alpha=1.0)
    ax_scale.axvline(0.0, color=PALETTE["fish_edge"], lw=1.0)
    ax_scale.axvline(hp.MUCUS_THICKNESS_UM, color=PALETTE["mucus_edge"], lw=1.2)
    ax_scale.annotate(
        "",
        xy=(0.0, 0.62),
        xytext=(hp.MUCUS_THICKNESS_UM, 0.62),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["accent"], lw=1.5),
    )
    ax_scale.text(0.5 * hp.MUCUS_THICKNESS_UM, 0.67, "600 μm", color=PALETTE["accent"], ha="center", va="bottom", fontweight="bold")
    ax_scale.text(-145.0, 0.22, "鱼体", color="white", ha="center", va="center", fontweight="bold")
    ax_scale.text(300.0, 0.22, "黏液层", color=PALETTE["neutral"], ha="center", va="center", fontweight="bold")
    ax_scale.text(750.0, 0.22, "水体", color=PALETTE["neutral"], ha="center", va="center")
    ax_scale.set_xlim(-300.0, 900.0)
    ax_scale.set_ylim(0.0, 1.0)
    ax_scale.set_yticks([])
    ax_scale.set_xlabel("距鱼体表面的法向距离（μm）")
    ax_scale.set_title("600 μm 黏液层定义")
    add_panel_label(ax_scale, "b")

    assert_axes_do_not_overlap(fig, "geometry")
    save_figure(fig, output_dir / "00_real_scale_fish_geometry")


def mask_full_hist_to_mucus(hist, x_edges, y_edges, hp: Params):
    xc = 0.5 * (x_edges[:-1] + x_edges[1:]) * 1000.0
    yc = 0.5 * (y_edges[:-1] + y_edges[1:]) * 1000.0
    X, Y = np.meshgrid(xc, yc)
    d = approximate_signed_distance(X, Y, hp)
    masked = np.array(hist, dtype=float)
    masked[(d <= 0.0) | (d > hp.MUCUS_THICKNESS_UM) | (masked <= 0.0)] = np.nan
    return masked


def plot_full_heatmap(results: Dict[str, dict], hp: Params, output_dir: Path):
    combined = sum(np.asarray(results[name]["full_hist"], dtype=float) for name in REGIONS)
    x_edges = np.asarray(results["head"]["full_x_edges"])
    y_edges = np.asarray(results["head"]["full_y_edges"])
    masked = mask_full_hist_to_mucus(combined, x_edges, y_edges, hp)
    finite = masked[np.isfinite(masked)]
    vmax = max(float(np.nanmax(finite)) if finite.size else 1.0, 2.0)

    fig = plt.figure(figsize=(11.8, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.035])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.pcolormesh(x_edges, y_edges, masked, shading="auto", cmap="magma", norm=LogNorm(vmin=1.0, vmax=vmax), rasterized=True)
    draw_fish(ax, hp, fill_mucus=False)

    for region_name, spec in REGIONS.items():
        xa, ya = region_arc(region_name, hp)
        ax.plot(xa, ya, lw=2.5, color=spec["color"], solid_capstyle="round", zorder=7)

    ax.text(-52.6, 3.2, "头部", color=PALETTE["head"], ha="right", va="center", fontweight="bold")
    ax.text(0.0, 17.0, "侧面", color=PALETTE["side"], ha="center", va="bottom", fontweight="bold")
    ax.text(52.6, 3.2, "后部", color=PALETTE["rear"], ha="left", va="center", fontweight="bold")

    flow_angle = np.deg2rad(hp.FLOW_ANGLE_DEG)
    start = np.array([-43.0, -13.8])
    delta = 9.0 * np.array([np.cos(flow_angle), np.sin(flow_angle)])
    ax.annotate("", xy=start + delta, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PALETTE["accent"]))
    ax.text(start[0] + 0.5 * delta[0], start[1] - 1.4, f"相对水流 {hp.FLOW_ANGLE_DEG:.0f}°", color=PALETTE["accent"], ha="center", va="top")

    ax.set_xlim(-55.0, 55.0)
    ax.set_ylim(-18.5, 18.5)
    ax.set_xlabel("鱼体纵轴位置（mm）")
    ax.set_ylabel("横向位置（mm）")
    ax.set_title("整鱼 600 μm 黏液层内的细菌占据热图")
    add_panel_label(ax, "a")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("黏液层内累计占据次数（对数色标）")

    assert_axes_do_not_overlap(fig, "full heatmap")
    save_figure(fig, output_dir / "01_whole_fish_mucus_heatmap")


def plot_local_heatmaps(results: Dict[str, dict], hp: Params, output_dir: Path):
    all_positive = []
    for name in REGIONS:
        h = np.asarray(results[name]["local_hist"], dtype=float)
        all_positive.append(h[h > 0.0])
    positive = np.concatenate([a for a in all_positive if a.size])
    vmax = max(float(np.max(positive)) if positive.size else 1.0, 2.0)

    fig = plt.figure(figsize=(13.2, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im = None
    for idx, (ax, region_name) in enumerate(zip(axes, REGIONS.keys())):
        spec = REGIONS[region_name]
        res = results[region_name]
        h = np.asarray(res["local_hist"], dtype=float)
        masked = np.ma.masked_where(h <= 0.0, h)
        x_edges = np.asarray(res["local_x_edges"])
        depth_edges = np.asarray(res["depth_edges"])

        im = ax.pcolormesh(x_edges, depth_edges, masked, shading="auto", cmap="magma", norm=LogNorm(vmin=1.0, vmax=vmax), rasterized=True)
        ax.axhline(0.0, color=PALETTE["mucus_edge"], lw=1.0)
        ax.axhline(hp.MUCUS_THICKNESS_UM, color=PALETTE["fish_edge"], lw=1.2)
        ax.fill_between(
            [x_edges[0], x_edges[-1]],
            hp.MUCUS_THICKNESS_UM,
            hp.MUCUS_THICKNESS_UM + 55.0,
            color=PALETTE["fish"],
            alpha=0.95,
            clip_on=False,
        )
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(hp.MUCUS_THICKNESS_UM + 55.0, 0.0)
        ax.set_xlabel("沿表面的切向位置（mm）")
        if idx == 0:
            ax.set_ylabel("从黏液外缘向鱼体的穿透深度（μm）")
            ax.text(x_edges[0] + 0.35, 22.0, "黏液外缘", color=PALETTE["mucus_edge"], ha="left", va="top")
            ax.text(x_edges[0] + 0.35, hp.MUCUS_THICKNESS_UM - 18.0, "鱼体表面", color="white", ha="left", va="bottom", fontweight="bold")
        else:
            ax.tick_params(labelleft=False)

        pct = 100.0 * float(res["penetration_fraction"])
        ax.set_title(f"{spec['label']}：600 μm 穿透热图\n进入鱼体 {pct:.1f}%（{res['penetrated_count']}/{res['n']}）", color=spec["color"], fontweight="bold")
        add_panel_label(ax, chr(ord("a") + idx))

    assert im is not None
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("累计细菌占据次数（对数色标）")

    assert_axes_do_not_overlap(fig, "local heatmaps")
    save_figure(fig, output_dir / "02_head_side_rear_600um_heatmaps")


def main():
    hp = HP
    output_dir = Path(hp.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    for i, region_name in enumerate(REGIONS):
        results[region_name] = simulate_region(region_name, hp, seed_offset=1000 * i)

    plot_geometry(hp, output_dir)
    plot_full_heatmap(results, hp, output_dir)
    plot_local_heatmaps(results, hp, output_dir)

    print(f"Fish length: {hp.FISH_LENGTH_MM:.1f} mm")
    print(f"Fish height: {hp.FISH_HEIGHT_MM:.1f} mm")
    print(f"Mucus thickness: {hp.MUCUS_THICKNESS_UM:.0f} um")
    for region_name, spec in REGIONS.items():
        res = results[region_name]
        print(
            f"{spec['label']}: penetration={100.0 * res['penetration_fraction']:.2f}% "
            f"({res['penetrated_count']}/{res['n']})"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
