#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blue-light uniformity model for an Archimedes-screw immersion reactor.
The model couples geometry, attenuating/scattering transport, moving screw
shadowing, and a YF1/FixJ Hill-response surrogate. It generates the numerical
results and figures used in the accompanying LaTeX report.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import lsq_linear
from scipy.stats import qmc, spearmanr

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# -------------------------- plotting style --------------------------
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
]
for fp in font_candidates:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams.update({
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# -------------------------- model parameters --------------------------
@dataclass
class Geometry:
    shell_inner_radius: float = 0.175   # m, CAD: 175 mm
    shell_outer_radius: float = 0.190   # m, CAD: 190 mm
    shell_length: float = 1.195         # m, CAD: 1195 mm
    shaft_radius: float = 0.030         # m, CAD: 30 mm
    guide_radius: float = 0.010         # m, CAD bore: 10 mm radius
    pitch: float = 0.145                # m, CAD: 145 mm
    turns: int = 6                      # inferred from helix arc length and CAD view
    blade_thickness: float = 0.003      # m, engineering assumption
    rpm: float = 3.0                    # nominal; parameterized in report

    @property
    def active_length(self) -> float:
        return self.pitch * self.turns

    @property
    def active_z0(self) -> float:
        return 0.5 * (self.shell_length - self.active_length)

    @property
    def active_z1(self) -> float:
        return self.active_z0 + self.active_length

    @property
    def helix_length(self) -> float:
        return self.turns * math.sqrt((2 * math.pi * self.shell_inner_radius) ** 2 + self.pitch ** 2)


@dataclass
class Optical:
    wavelength_nm: float = 470.0
    mu_a: float = 1.0                  # m^-1, nominal calibration parameter
    mu_s: float = 3.0                  # m^-1, nominal calibration parameter
    anisotropy_g: float = 0.85
    wall_transmission: float = 0.88
    shaft_transmission: float = 0.76
    diffuse_fraction: float = 0.65
    lambert_order: float = 0.7
    shadow_depth_outer: float = 0.64
    shadow_depth_inner: float = 0.76
    shadow_sigma_deg: float = 10.0
    reflection_gain: float = 1.10      # diffuse reflector / multiple-scattering gain

    @property
    def mu_t(self) -> float:
        return self.mu_a + self.mu_s

    @property
    def mu_sp(self) -> float:
        return self.mu_s * (1.0 - self.anisotropy_g)

    @property
    def mu_eff(self) -> float:
        return math.sqrt(3.0 * self.mu_a * (self.mu_a + self.mu_sp))


@dataclass
class Bio:
    half_saturation_lux: float = 15.0  # previous Model 3
    hill_n: float = 2.0
    design_safety_factor: float = 3.0  # n=2 -> 90% response at 3*E50
    residence_time_s: float = 120.0


def cie_v_lambda(wavelength_nm: float) -> float:
    arr = np.loadtxt(DATA / "CIE_sle_photopic.csv", delimiter=",")
    return float(np.interp(wavelength_nm, arr[:, 0], arr[:, 1]))


def lux_to_wm2(lux: float, wavelength_nm: float) -> float:
    v = cie_v_lambda(wavelength_nm)
    return lux / (683.0 * v)


# -------------------------- geometry sampling --------------------------
def wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def make_points(g: Geometry, nr: int = 6, nt: int = 24, nz: int = 24) -> Dict[str, np.ndarray]:
    # Avoid exact boundaries to reduce point-source and wall singularities.
    r = np.linspace(g.shaft_radius + 0.012, g.shell_inner_radius - 0.010, nr)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    z = np.linspace(g.active_z0 + 0.02, g.active_z1 - 0.02, nz)
    rr, tt, zz = np.meshgrid(r, th, z, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    pts = np.column_stack([x.ravel(), y.ravel(), zz.ravel()])
    return {"xyz": pts, "r": rr.ravel(), "theta": tt.ravel(), "z": zz.ravel(),
            "shape": np.array([nr, nt, nz], dtype=int)}


def source_layout(g: Geometry, ntheta: int = 8, nzones: int = 8) -> Dict[str, np.ndarray]:
    zc = np.linspace(g.active_z0 + g.active_length/(2*nzones),
                     g.active_z1 - g.active_length/(2*nzones), nzones)
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    Rled = g.shell_outer_radius + 0.014
    outer_pos = []
    outer_n = []
    outer_zone = []
    for iz, z in enumerate(zc):
        for t in th:
            outer_pos.append([Rled*np.cos(t), Rled*np.sin(t), z])
            outer_n.append([-np.cos(t), -np.sin(t), 0.0])
            outer_zone.append(iz)
    # Central side-emitting guide is represented by one isotropic segment per zone.
    inner_pos = np.column_stack([np.zeros(nzones), np.zeros(nzones), zc])
    return {
        "outer_pos": np.asarray(outer_pos),
        "outer_n": np.asarray(outer_n),
        "outer_zone": np.asarray(outer_zone, dtype=int),
        "inner_pos": inner_pos,
        "z_centers": zc,
        "theta_sources": th,
        "n_theta": np.array([ntheta]),
        "n_zones": np.array([nzones]),
    }


def shadow_visibility(theta: np.ndarray, z: np.ndarray, phase: float, g: Geometry,
                      depth: float, sigma_deg: float) -> np.ndarray:
    # Local blade plane of a single-start helix. The moving blade shadow rotates with phase.
    blade_angle = 2*np.pi*(z - g.active_z0)/g.pitch + phase
    dtheta = wrap_angle(theta - blade_angle)
    sigma = np.deg2rad(sigma_deg)
    return 1.0 - depth*np.exp(-0.5*(dtheta/sigma)**2)


def group_matrix(points: Dict[str, np.ndarray], layout: Dict[str, np.ndarray],
                 g: Geometry, o: Optical, phase: float) -> np.ndarray:
    """Return irradiance matrix A [n_points, 2*n_zones] per watt optical.

    First n_zones are total power of the 8 symmetric external bars in each axial zone;
    last n_zones are power extracted by the central guide in each axial zone.
    """
    X = points["xyz"]
    npts = len(X)
    nzones = int(layout["n_zones"][0])
    ntheta = int(layout["n_theta"][0])
    A = np.zeros((npts, 2*nzones), dtype=float)

    # Moving shadow is applied only to ballistic component; diffuse component survives.
    vis_o = shadow_visibility(points["theta"], points["z"], phase, g,
                              o.shadow_depth_outer, o.shadow_sigma_deg)
    vis_i = shadow_visibility(points["theta"], points["z"], phase + np.pi/2, g,
                              o.shadow_depth_inner, o.shadow_sigma_deg*1.1)

    # External bars
    for s, nvec, iz in zip(layout["outer_pos"], layout["outer_n"], layout["outer_zone"]):
        V = X - s
        d = np.linalg.norm(V, axis=1)
        u = V / d[:, None]
        cos_emit = np.clip(u @ nvec, 0.0, 1.0)
        direct_geom = (o.lambert_order + 1.0)/(2*np.pi) * cos_emit**o.lambert_order / (d*d)
        diffuse_geom = 1.0/(4*np.pi*(d*d + 0.010**2))
        direct = (1-o.diffuse_fraction)*direct_geom*np.exp(-o.mu_t*d)*vis_o
        diffuse = o.diffuse_fraction*diffuse_geom*np.exp(-o.mu_eff*d)
        # iz variable denotes total zone power across all bars, hence divide per bar.
        A[:, iz] += o.wall_transmission*o.reflection_gain*(direct + diffuse)/ntheta

    # Central side-emitting guide
    for iz, s in enumerate(layout["inner_pos"]):
        V = X - s
        d = np.linalg.norm(V, axis=1)
        # finite source radius regularization and approximately cylindrical extraction
        d2 = d*d + 0.014**2
        direct_geom = 1.0/(4*np.pi*d2)
        diffuse_geom = 1.0/(4*np.pi*(d2 + 0.015**2))
        direct = (1-o.diffuse_fraction)*direct_geom*np.exp(-o.mu_t*d)*vis_i
        diffuse = o.diffuse_fraction*diffuse_geom*np.exp(-o.mu_eff*d)
        A[:, nzones+iz] = o.shaft_transmission*o.reflection_gain*(direct + diffuse)
    return A


def phase_matrices(points: Dict[str, np.ndarray], layout: Dict[str, np.ndarray],
                   g: Geometry, o: Optical, nphase: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    phases = np.linspace(0, 2*np.pi, nphase, endpoint=False)
    mats = np.stack([group_matrix(points, layout, g, o, ph) for ph in phases], axis=0)
    return phases, mats


def optimize_power(Aavg: np.ndarray, nzones: int, target: float = 1.0) -> np.ndarray:
    npts, nvar = Aavg.shape
    assert nvar == 2*nzones
    # Smoothness matrices for manufacturable axial power/extraction profiles.
    D = np.zeros((2*(nzones-1), nvar))
    for i in range(nzones-1):
        D[i, i] = -1; D[i, i+1] = 1
        D[nzones-1+i, nzones+i] = -1; D[nzones-1+i, nzones+i+1] = 1
    reg = np.eye(nvar)
    # Slight penalty on central-guide power to limit thermal loading inside the shaft.
    reg[nzones:, nzones:] *= 1.5
    M = np.vstack([Aavg, 0.06*D, 0.008*reg])
    b = np.concatenate([np.full(npts, target), np.zeros(D.shape[0] + nvar)])
    lb = np.zeros(nvar)
    ub = np.r_[np.full(nzones, 4.0), np.full(nzones, 1.5)]
    sol = lsq_linear(M, b, bounds=(lb, ub), lsmr_tol="auto", max_iter=500)
    return sol.x


def biological_response(E: np.ndarray, E50: float, n: float) -> np.ndarray:
    return E**n / (E**n + E50**n)


def metrics(Ephase: np.ndarray, E50: float, n_hill: float) -> Dict[str, float]:
    # Ephase [nphase, npoints]
    Eavg = Ephase.mean(axis=0)
    Bphase = biological_response(Ephase, E50, n_hill)
    Bavg = Bphase.mean(axis=0)
    phase_cv = Ephase.std(axis=1)/Ephase.mean(axis=1)
    return {
        "mean_E": float(Eavg.mean()),
        "min_E": float(Eavg.min()),
        "max_E": float(Eavg.max()),
        "cv_E_avg": float(Eavg.std()/Eavg.mean()),
        "uniformity_min_mean": float(Eavg.min()/Eavg.mean()),
        "max_min_ratio": float(Eavg.max()/Eavg.min()),
        "instantaneous_cv_mean": float(phase_cv.mean()),
        "instantaneous_cv_max": float(phase_cv.max()),
        "bio_mean": float(Bavg.mean()),
        "bio_min": float(Bavg.min()),
        "bio_cv": float(Bavg.std()/Bavg.mean()),
    }


def scale_for_bio_min(Ephase_unit: np.ndarray, E50: float, n_hill: float, target: float = 0.90) -> float:
    """Find the smallest scalar that makes the minimum phase-averaged Hill response reach target."""
    lo, hi = 0.0, 1.0
    def f(s: float) -> float:
        B = biological_response(Ephase_unit*s, E50, n_hill).mean(axis=0)
        return float(B.min())
    while f(hi) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Unable to reach biological target")
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def baseline_power(Aavg_4bar: np.ndarray, nzones: int, min_target: float) -> np.ndarray:
    # Equal axial power, external only. Scale to the requested minimum dose.
    p = np.r_[np.ones(nzones), np.zeros(nzones)]
    E = Aavg_4bar @ p
    p *= min_target/max(E.min(), 1e-12)
    return p


def summary_run() -> Dict[str, object]:
    g = Geometry()
    o = Optical()
    b = Bio()
    E50 = lux_to_wm2(b.half_saturation_lux, o.wavelength_nm)
    Emin_design = b.design_safety_factor * E50

    points = make_points(g)
    ntheta_opt, nzones = 12, 10
    layout = source_layout(g, ntheta=ntheta_opt, nzones=nzones)
    phases, mats = phase_matrices(points, layout, g, o, nphase=18)
    Aavg = mats.mean(axis=0)
    p_opt = optimize_power(Aavg, nzones, target=1.0)
    Ephase_opt_unit = np.einsum("tpv,v->tp", mats, p_opt)
    p_opt *= scale_for_bio_min(Ephase_opt_unit, E50, b.hill_n, target=0.90)
    Ephase_opt = np.einsum("tpv,v->tp", mats, p_opt)
    met_opt = metrics(Ephase_opt, E50, b.hill_n)

    # Baseline: four bars only, equal zone powers. Scale to the same minimum biological response.
    layout4 = source_layout(g, ntheta=4, nzones=nzones)
    ph4, mats4 = phase_matrices(points, layout4, g, o, nphase=18)
    Aavg4 = mats4.mean(axis=0)
    p_base = np.r_[np.ones(nzones), np.zeros(nzones)]
    Ephase_base_unit = np.einsum("tpv,v->tp", mats4, p_base)
    p_base *= scale_for_bio_min(Ephase_base_unit, E50, b.hill_n, target=0.90)
    Ephase_base = np.einsum("tpv,v->tp", mats4, p_base)
    met_base = metrics(Ephase_base, E50, b.hill_n)

    result = {
        "geometry": asdict(g),
        "optical": {**asdict(o), "mu_t": o.mu_t, "mu_sp": o.mu_sp, "mu_eff": o.mu_eff},
        "bio": {**asdict(b), "cie_v_lambda": cie_v_lambda(o.wavelength_nm),
                "E50_W_m2": E50, "Emin_design_W_m2": Emin_design},
        "optimized_power_W": p_opt.tolist(),
        "baseline_power_W": p_base.tolist(),
        "layout": {"optimized_outer_bars": ntheta_opt, "optimized_axial_zones": nzones,
                   "baseline_outer_bars": 4, "central_guide_zones": nzones},
        "optimized_total_optical_W": float(p_opt.sum()),
        "baseline_total_optical_W": float(p_base.sum()),
        "optimized_metrics": met_opt,
        "baseline_metrics": met_base,
    }

    np.savez_compressed(DATA / "simulation_arrays.npz",
                        points_xyz=points["xyz"], points_r=points["r"],
                        points_theta=points["theta"], points_z=points["z"],
                        shape=points["shape"], phases=phases,
                        Ephase_opt=Ephase_opt, Ephase_base=Ephase_base,
                        p_opt=p_opt, p_base=p_base, z_centers=layout["z_centers"])
    (DATA / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(g, o, b, points, layout, phases, Ephase_opt, Ephase_base, p_opt, p_base, E50)
    sensitivity_analysis(g, o, b, p_opt, E50)
    return result


# -------------------------- figures --------------------------
def make_figures(g: Geometry, o: Optical, b: Bio, points: Dict[str, np.ndarray],
                 layout: Dict[str, np.ndarray], phases: np.ndarray,
                 Eopt: np.ndarray, Ebase: np.ndarray, p_opt: np.ndarray,
                 p_base: np.ndarray, E50: float) -> None:
    # 1. Cross-section geometry and source arrangement
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.set_aspect("equal")
    th = np.linspace(0, 2*np.pi, 400)
    for r, ls, lw, label in [
        (g.shell_outer_radius, "-", 2.2, "外筒半径 190 mm"),
        (g.shell_inner_radius, "-", 2.0, "内腔半径 175 mm"),
        (g.shaft_radius, "-", 2.0, "螺杆轴半径 30 mm"),
        (g.guide_radius, "--", 1.8, "中心光导半径 10 mm"),
    ]:
        ax.plot(r*np.cos(th), r*np.sin(th), ls, lw=lw, label=label)
    # blade at one phase
    ax.plot([g.shaft_radius, g.shell_inner_radius], [0, 0], lw=4, alpha=.65, label="螺旋叶片截面")
    # LED bars around circumference
    for k, ang in enumerate(layout["theta_sources"]):
        rr = g.shell_outer_radius + 0.014
        x, y = rr*np.cos(ang), rr*np.sin(ang)
        ax.scatter(x, y, s=115, marker="s", edgecolor="black", linewidth=.6,
                   label="外置 470 nm LED 条" if k == 0 else None)
        ax.arrow(x, y, -0.023*np.cos(ang), -0.023*np.sin(ang), width=.0008,
                 head_width=.008, length_includes_head=True)
    ax.scatter(0, 0, s=70, marker="o", edgecolor="black", linewidth=.6,
               label="固定式侧发光光导")
    ax.annotate("双侧供光：外向内 + 中心向外", xy=(0.02, -0.02), xytext=(0.08, -0.11),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    ax.set_xlim(-0.24, 0.24); ax.set_ylim(-0.24, 0.24)
    ax.set_xlabel("横向坐标 x / m"); ax.set_ylabel("纵向坐标 y / m")
    ax.set_title("优化方案横截面：12 方位外置 LED + 轴心侧发光光导")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(FIG / "geometry_cross_section.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Unwrapped layout with helix and axial zones
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    z0, z1 = g.active_z0, g.active_z1
    for ths in np.rad2deg(layout["theta_sources"]):
        ax.plot([z0, z1], [ths, ths], lw=3, alpha=.65)
    zline = np.linspace(z0, z1, 1200)
    helix = np.mod(np.rad2deg(2*np.pi*(zline-z0)/g.pitch), 360)
    # break wrapped line for clean plotting
    jump = np.where(np.abs(np.diff(helix)) > 180)[0] + 1
    for seg in np.split(np.arange(len(zline)), jump):
        ax.plot(zline[seg], helix[seg], "k--", lw=1.4)
    for zc in layout["z_centers"]:
        ax.axvline(zc, color="gray", lw=.6, alpha=.4)
    ax.set_xlim(z0, z1); ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 45))
    ax.set_xlabel("轴向位置 z / m"); ax.set_ylabel("展开圆周角 / °")
    ax.set_title("圆筒展开图：LED 方位与螺旋叶片阴影轨迹错位布置")
    ax.text(z0+0.01, 337, "黑虚线：叶片在一个转角下的投影\n彩色直线：12 条纵向 LED",
            va="top", bbox=dict(boxstyle="round,pad=.3", fc="white", alpha=.9))
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(FIG / "unwrapped_layout.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Irradiance maps at outer representative radius, averaged over time and radius band
    nr, nt, nz = points["shape"]
    def map_theta_z(Ephase):
        Eavg = Ephase.mean(axis=0).reshape(nr, nt, nz)
        # average the three middle/outer radial shells; these are most relevant for fluid bulk
        return Eavg[2:, :, :].mean(axis=0)
    mb = map_theta_z(Ebase)
    mo = map_theta_z(Eopt)
    Bbase = biological_response(mb, E50, b.hill_n)
    Bopt = biological_response(mo, E50, b.hill_n)
    extent = [g.active_z0, g.active_z1, 0, 360]
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)
    vmin = min(mb.min(), mo.min()); vmax = max(mb.max(), mo.max())
    im0 = axs[0,0].imshow(mb, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    axs[0,0].set_title("基线：4 条外置 LED，时间平均辐照度")
    im1 = axs[0,1].imshow(mo, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("优化：12 条外置 LED + 中心光导")
    im2 = axs[1,0].imshow(Bbase, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axs[1,0].set_title("基线：YF1 蓝光响应分数")
    im3 = axs[1,1].imshow(Bopt, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axs[1,1].set_title("优化：YF1 蓝光响应分数")
    for ax in axs.ravel():
        ax.set_xlabel("z / m"); ax.set_ylabel("圆周角 / °")
    cb1 = fig.colorbar(im1, ax=axs[0,:], shrink=.88, pad=.02)
    cb1.set_label("辐照度 / W m$^{-2}$")
    cb2 = fig.colorbar(im3, ax=axs[1,:], shrink=.88, pad=.02)
    cb2.set_label("归一化生物响应")
    fig.savefig(FIG / "uniformity_maps.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Radial profiles and phase CV
    rr = np.unique(points["r"])
    Eo_avg = Eopt.mean(axis=0).reshape(nr, nt, nz)
    Eb_avg = Ebase.mean(axis=0).reshape(nr, nt, nz)
    radial_o = Eo_avg.mean(axis=(1,2)); radial_b = Eb_avg.mean(axis=(1,2))
    radial_o_min = Eo_avg.min(axis=(1,2)); radial_b_min = Eb_avg.min(axis=(1,2))
    cv_o = Eopt.std(axis=1)/Eopt.mean(axis=1)
    cv_b = Ebase.std(axis=1)/Ebase.mean(axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    axs[0].plot(rr*1000, radial_b, "o-", label="基线均值")
    axs[0].plot(rr*1000, radial_b_min, "o--", label="基线最小值")
    axs[0].plot(rr*1000, radial_o, "s-", label="优化均值")
    axs[0].plot(rr*1000, radial_o_min, "s--", label="优化最小值")
    axs[0].axhline(3*E50, color="black", ls=":", lw=1.4, label="设计下限 3E$_{50}$")
    axs[0].set_xlabel("半径 r / mm"); axs[0].set_ylabel("时间平均辐照度 / W m$^{-2}$")
    axs[0].set_title("径向均匀性：中心光导补偿内层暗区")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.25)
    axs[1].plot(np.rad2deg(phases), cv_b*100, "o-", label="基线")
    axs[1].plot(np.rad2deg(phases), cv_o*100, "s-", label="优化")
    axs[1].set_xlabel("螺杆转角 / °"); axs[1].set_ylabel("瞬时空间 CV / %")
    axs[1].set_title("转动阴影下的瞬时波动")
    axs[1].legend(); axs[1].grid(alpha=.25)
    fig.savefig(FIG / "radial_phase_profiles.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Optimized powers
    nzones = len(layout["z_centers"])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(1, nzones+1)
    w = .36
    ax.bar(x-w/2, p_opt[:nzones], width=w, label="外置 12 条 LED 的该轴区总光功率")
    ax.bar(x+w/2, p_opt[nzones:], width=w, label="中心光导该轴区提取光功率")
    ax.plot(x, p_base[:nzones], "k--o", lw=1.3, ms=4, label="4 条 LED 基线总功率")
    ax.set_xlabel("轴向分区（入口 → 出口）"); ax.set_ylabel("光学功率 / W")
    ax.set_title("优化得到的轴向功率配方：端部补偿 + 中心双向照明")
    ax.set_xticks(x); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / "optimized_power_profile.png", bbox_inches="tight")
    plt.close(fig)


def sensitivity_analysis(g: Geometry, o0: Optical, b: Bio, p_opt: np.ndarray, E50: float) -> None:
    """Uncertainty propagation and rank sensitivity using a compact grid."""
    rng = np.random.default_rng(20260723)
    points = make_points(g, nr=4, nt=12, nz=10)
    layout = source_layout(g, ntheta=12, nzones=10)
    sampler = qmc.LatinHypercube(d=6, seed=rng)
    N = 240
    U = sampler.random(N)
    ranges = np.array([
        [0.5, 2.0],    # mu_a
        [1.0, 6.0],    # mu_s
        [0.75, 0.93],  # g
        [0.80, 0.95],  # wall transmission
        [0.35, 0.80],  # shadow depth
        [0.95, 1.20],  # reflection gain
    ])
    X = qmc.scale(U, ranges[:,0], ranges[:,1])
    y_cv = np.zeros(N); y_umin = np.zeros(N); y_bmin = np.zeros(N)
    phases = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for i, row in enumerate(X):
        oo = Optical(mu_a=row[0], mu_s=row[1], anisotropy_g=row[2],
                     wall_transmission=row[3], shadow_depth_outer=row[4],
                     shadow_depth_inner=min(0.92, row[4]+0.12), reflection_gain=row[5])
        mats = np.stack([group_matrix(points, layout, g, oo, ph) for ph in phases])
        E = np.einsum("tpv,v->tp", mats, p_opt)
        met = metrics(E, E50, b.hill_n)
        y_cv[i] = met["cv_E_avg"]
        y_umin[i] = met["uniformity_min_mean"]
        y_bmin[i] = met["bio_min"]
    names = ["吸收系数 μa", "散射系数 μs", "各向异性 g", "筒壁透过率 T", "叶片阴影深度", "漫反射增益"]
    rho_cv = np.array([spearmanr(X[:,j], y_cv).statistic for j in range(X.shape[1])])
    rho_u = np.array([spearmanr(X[:,j], y_umin).statistic for j in range(X.shape[1])])
    rho_b = np.array([spearmanr(X[:,j], y_bmin).statistic for j in range(X.shape[1])])
    out = {
        "parameter_names": names,
        "ranges": ranges.tolist(),
        "spearman_cv": rho_cv.tolist(),
        "spearman_uniformity": rho_u.tolist(),
        "spearman_bio_min": rho_b.tolist(),
        "cv_quantiles": np.quantile(y_cv, [0.05,0.5,0.95]).tolist(),
        "uniformity_quantiles": np.quantile(y_umin, [0.05,0.5,0.95]).tolist(),
        "bio_min_quantiles": np.quantile(y_bmin, [0.05,0.5,0.95]).tolist(),
    }
    (DATA / "sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    order = np.arange(len(names))
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    axs[0].barh(order-0.18, rho_cv, height=.34, label="对空间 CV 的相关")
    axs[0].barh(order+0.18, rho_u, height=.34, label="对最小/平均均匀度的相关")
    axs[0].axvline(0, color="black", lw=.8)
    axs[0].set_yticks(order, names); axs[0].invert_yaxis()
    axs[0].set_xlabel("Spearman 等级相关系数")
    axs[0].set_title("参数不确定性敏感度")
    axs[0].legend(fontsize=8); axs[0].grid(axis="x", alpha=.2)
    axs[1].hist(y_bmin, bins=22, alpha=.8)
    q5, q50, q95 = np.quantile(y_bmin, [0.05,0.5,0.95])
    for q, lab in [(q5,"5%"),(q50,"中位数"),(q95,"95%")]:
        axs[1].axvline(q, ls="--", lw=1.2, label=f"{lab}: {q:.3f}")
    axs[1].set_xlabel("全域最小 YF1 响应分数")
    axs[1].set_ylabel("样本数")
    axs[1].set_title("240 组光学参数不确定性传播")
    axs[1].legend(fontsize=8); axs[1].grid(axis="y", alpha=.2)
    fig.savefig(FIG / "sensitivity_uncertainty.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    res = summary_run()
    print(json.dumps(res, ensure_ascii=False, indent=2))
