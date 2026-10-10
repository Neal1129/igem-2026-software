"""圆形培养皿 48 h 细菌运动、扩散与增殖模拟。

基于 merged_abm_full.py 的 run-and-tumble ABM 思路改写：
将微流控通道换成圆形培养皿，并从培养皿中心直接接种菌液。
运行：python3 circular_petri_dish_abm_48h.py
输出：output/petri_dish_48h_dark_vs_light.jpg（600 ppi）
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
    }
)


@dataclass
class HyperParams:
    # ---------------- 可直接调节的接种浓度超参数 ----------------
    BACTERIAL_CONCENTRATION_CFU_PER_ML: float = 1.0e8
    INOCULATION_VOLUME_UL: float = 10.0
    CFU_PER_AGENT: float = 1.0e2

    # ---------------- 培养皿与模拟时长 ----------------
    PETRI_DIAMETER_MM: float = 90.0
    TOTAL_HOURS: float = 48.0
    DT_MIN: float = 2.0
    SNAPSHOT_HOURS: tuple = (0, 12, 24, 36, 48)
    INOCULUM_SIGMA_MM: float = 0.8
    SEED: int = 20260823

    # ---------------- 扩散、游动与增殖 ----------------
    ROTATIONAL_DIFFUSION_PER_H: float = 1.5
    PASSIVE_DIFFUSION_MM2_PER_H: float = 0.015
    CARRYING_CAPACITY_CFU: float = 2.0e9

    DARK_SWIM_SPEED_MM_PER_H: float = 2.0
    DARK_TUMBLE_RATE_PER_H: float = 25.0
    DARK_GROWTH_RATE_PER_H: float = 0.063

    LIGHT_SWIM_SPEED_MM_PER_H: float = 2.55
    LIGHT_TUMBLE_RATE_PER_H: float = 20.0
    LIGHT_GROWTH_RATE_PER_H: float = 0.068

    # ---------------- 图片输出 ----------------
    DENSITY_BINS: int = 240
    SMOOTHING_PASSES: int = 12
    OUTPUT_DPI: int = 600


HP = HyperParams()
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def smooth_density(density: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        padded = np.pad(density, 1)
        density = (
            4 * padded[1:-1, 1:-1]
            + 2 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
            + padded[:-2, :-2]
            + padded[:-2, 2:]
            + padded[2:, :-2]
            + padded[2:, 2:]
        ) / 16
    return density


def reflect_at_dish_wall(x: np.ndarray, y: np.ndarray, radius: float) -> None:
    distance = np.sqrt(x * x + y * y)
    outside = distance > radius
    reflection_scale = (2 * radius - distance[outside]) / distance[outside]
    x[outside] *= reflection_scale
    y[outside] *= reflection_scale


def simulate_condition(
    hp: HyperParams,
    swim_speed: float,
    tumble_rate: float,
    growth_rate: float,
) -> dict:
    rng = np.random.default_rng(hp.SEED)
    radius = hp.PETRI_DIAMETER_MM / 2
    initial_cfu = hp.BACTERIAL_CONCENTRATION_CFU_PER_ML * hp.INOCULATION_VOLUME_UL * 1e-3
    n_agents = int(initial_cfu / hp.CFU_PER_AGENT)
    agent_cfu = np.full(n_agents, initial_cfu / n_agents)
    x = rng.normal(0, hp.INOCULUM_SIGMA_MM, n_agents)
    y = rng.normal(0, hp.INOCULUM_SIGMA_MM, n_agents)
    theta = rng.uniform(0, 2 * np.pi, n_agents)

    dt_h = hp.DT_MIN / 60
    total_steps = int(hp.TOTAL_HOURS / dt_h)
    snapshot_steps = {round(hour / dt_h): hour for hour in hp.SNAPSHOT_HOURS}
    snapshots = {}

    for step in range(total_steps + 1):
        if step in snapshot_steps:
            hour = snapshot_steps[step]
            snapshots[hour] = (x.copy(), y.copy(), agent_cfu.copy(), float(agent_cfu.sum()))

        if step == total_steps:
            break

        tumble = rng.random(n_agents) < tumble_rate * dt_h
        theta[tumble] = rng.uniform(0, 2 * np.pi, tumble.sum())
        theta += np.sqrt(2 * hp.ROTATIONAL_DIFFUSION_PER_H * dt_h) * rng.standard_normal(n_agents)
        x += swim_speed * np.cos(theta) * dt_h
        y += swim_speed * np.sin(theta) * dt_h
        x += np.sqrt(2 * hp.PASSIVE_DIFFUSION_MM2_PER_H * dt_h) * rng.standard_normal(n_agents)
        y += np.sqrt(2 * hp.PASSIVE_DIFFUSION_MM2_PER_H * dt_h) * rng.standard_normal(n_agents)
        reflect_at_dish_wall(x, y, radius)

        total_cfu = agent_cfu.sum()
        agent_cfu *= np.exp(growth_rate * dt_h * (1 - total_cfu / hp.CARRYING_CAPACITY_CFU))

    return snapshots


def density_image(snapshot: tuple, hp: HyperParams) -> np.ndarray:
    x, y, agent_cfu, _ = snapshot
    radius = hp.PETRI_DIAMETER_MM / 2
    edges = np.linspace(-radius, radius, hp.DENSITY_BINS + 1)
    density, _, _ = np.histogram2d(y, x, bins=(edges, edges), weights=agent_cfu)
    density = smooth_density(density, hp.SMOOTHING_PASSES)
    centers = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(centers, centers)
    density[xx * xx + yy * yy > radius * radius] = np.nan
    return density


def save_timecourse_figure(dark: dict, light: dict, hp: HyperParams) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    radius = hp.PETRI_DIAMETER_MM / 2
    conditions = [
        ("No light", dark, "#475569"),
        ("Blue light ON", light, "#0ea5e9"),
    ]
    density_maps = {
        label: {hour: density_image(snapshots[hour], hp) for hour in hp.SNAPSHOT_HOURS}
        for label, snapshots, _ in conditions
    }
    peak = max(np.nanmax(density) for rows in density_maps.values() for density in rows.values())
    for rows in density_maps.values():
        for density in rows.values():
            density[np.isfinite(density) & (density == 0)] = peak * 1e-6
    norm = LogNorm(vmin=peak * 1e-4, vmax=peak)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#f8fafc")
    cmap.set_bad(alpha=0)

    fig = plt.figure(figsize=(15.6, 6.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        len(hp.SNAPSHOT_HOURS) + 1,
        width_ratios=[1, 1, 1, 1, 1, 0.055],
        left=0.075,
        right=0.955,
        top=0.82,
        bottom=0.10,
        wspace=0.15,
        hspace=0.28,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[row, col]) for col in range(len(hp.SNAPSHOT_HOURS))]
            for row in range(2)
        ]
    )
    colorbar_ax = fig.add_subplot(grid[:, -1])
    for row, (label, snapshots, accent) in enumerate(conditions):
        for col, hour in enumerate(hp.SNAPSHOT_HOURS):
            ax = axes[row, col]
            ax.set_facecolor("#f8fafc")
            image = ax.imshow(
                density_maps[label][hour],
                origin="lower",
                extent=(-radius, radius, -radius, radius),
                cmap=cmap,
                norm=norm,
                interpolation="bicubic",
            )
            dish = Circle((0, 0), radius, fill=False, edgecolor=accent, linewidth=2.4)
            ax.add_patch(dish)
            image.set_clip_path(dish)
            ax.set_aspect("equal")
            ax.set_xlim(-radius * 1.06, radius * 1.06)
            ax.set_ylim(-radius * 1.06, radius * 1.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"{hour} h", fontsize=11, weight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(
                    label,
                    color=accent,
                    fontsize=10,
                    weight="bold",
                    labelpad=14,
                )
            population = snapshots[hour][3]
            ax.set_xlabel(
                f"{population:.2e} CFU",
                color="#0f172a",
                fontsize=8,
                labelpad=3,
            )

    fig.suptitle(
        f"Central inoculation and bacterial spreading in a circular Petri dish (0–48 h)\n"
        f"Inoculum = {hp.BACTERIAL_CONCENTRATION_CFU_PER_ML:.1e} CFU/mL × {hp.INOCULATION_VOLUME_UL:g} µL",
        fontsize=14,
        weight="bold",
        y=0.965,
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
    )
    colorbar.set_label("Bacterial abundance\n(CFU per grid cell, log scale)", fontsize=8.5, labelpad=8)
    colorbar.ax.tick_params(labelsize=7.5, length=3)

    output_path = OUTPUT_DIR / "petri_dish_48h_dark_vs_light.jpg"
    fig.savefig(output_path, dpi=hp.OUTPUT_DPI, format="jpg", pil_kwargs={"quality": 95})
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    dark_snapshots = simulate_condition(
        HP,
        HP.DARK_SWIM_SPEED_MM_PER_H,
        HP.DARK_TUMBLE_RATE_PER_H,
        HP.DARK_GROWTH_RATE_PER_H,
    )
    light_snapshots = simulate_condition(
        HP,
        HP.LIGHT_SWIM_SPEED_MM_PER_H,
        HP.LIGHT_TUMBLE_RATE_PER_H,
        HP.LIGHT_GROWTH_RATE_PER_H,
    )
    output = save_timecourse_figure(dark_snapshots, light_snapshots, HP)
    print(f"Saved 600 ppi JPG: {output}")
