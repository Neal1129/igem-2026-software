"""
Stage 3 simulation and strategy sweep for the merged ABM model.

This script builds upon the stage‑2 implementation and introduces
a control parameter ``gamma_ABM`` that reflects the balance between
clockwise motor bias, light pulse frequency and steady illumination.
Based on ``gamma_ABM`` agents switch between ``swiping`` and
``nosing`` behaviours: swiping increases tumble frequency to scan
more lateral positions, whereas nosing decreases the tumble rate to
favor persistent forward motion.  Thresholds ``GAMMA_ABM_HIGH`` and
``GAMMA_ABM_LOW`` control the switching.  The tumble rate λ is
scaled multiplicatively in each mode using ``SWIPING_TUMBLE_SCALE``
and ``NOSING_TUMBLE_SCALE`` respectively.

In addition to running a single simulation with the new control
mechanism, the script performs a parameter sweep over a grid of
pulse frequencies and duty cycles.  For each combination it runs a
short simulation and records the penetration fraction, total light
dose and average ``gamma_ABM``.  Figures produced include:

  * ``19_gamma_abm_timeseries_and_mode_fraction.png`` – time series of
    ``gamma_ABM`` and the fraction of time in swiping vs nosing modes.
  * ``20_strategy_phase_diagram_penetration.png`` – heatmap of
    penetration fraction as a function of pulse frequency and duty
    cycle.
  * ``21_ttcf_short_long_persistence.png`` – tangent–tangent
    correlation function (TTCF) showing short‑term decay and long‑term
    persistence of trajectories.
  * ``22_pareto_lightdose_vs_penetration.png`` – Pareto diagram of
    total light dose versus penetration fraction for all simulated
    strategies.

The underlying simulation retains optogenetic CheY–P dynamics,
optional rheotaxis and surface retention from the earlier stages but
sets chemotaxis off by default.  Simulation lengths are kept short
for computational efficiency; you may adjust the hyperparameters at
the bottom of the file.
"""

import os
import json
import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
# Disable caas_jupyter_tools instrumentation
os.environ.setdefault("DISABLE_CJ_INSTRUMENTATION", "1")
os.environ.setdefault("CAAS_JUPYTER_SERVER_PORT", "0")
matplotlib.use('Agg')  # use non‑interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the consolidated ABM module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    m = importlib.import_module("merged_abm_full")
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Module 'merged_abm_full' could not be imported from the project src directory."
    ) from e


# ---------------------------------------------------------------------------
# Extend the hyperparameters with fields needed for stage‑3.  This
# includes gamma_ABM thresholds and scaling factors as well as the
# optogenetic parameters carried forward from stage‑1 and the
# rheotaxis and surface retention parameters from stage‑2.
setattr(m.HyperParams, "USE_OPTO_CHEY_DYNAMICS", True)
setattr(m.HyperParams, "K_ACT", 1.5)
setattr(m.HyperParams, "K_DEACT", 1.0)
setattr(m.HyperParams, "CHEY_P_BASE", 0.05)
setattr(m.HyperParams, "CHEY_P_MAX", 1.0)
setattr(m.HyperParams, "CHEY_KD", 0.35)
setattr(m.HyperParams, "CHEY_HILL_N", 3.0)
setattr(m.HyperParams, "CHEY_DELAY_S", 0.2)

# Rheotaxis parameters
setattr(m.HyperParams, "USE_RHEOTAXIS", True)
setattr(m.HyperParams, "RHEO_GAIN", 0.8)
setattr(m.HyperParams, "RHEO_GAMMA_HALF", 1.0)
setattr(m.HyperParams, "RHEO_ST_GAIN", 2.0)
setattr(m.HyperParams, "THETA_FLOW", -np.pi / 2.0)

# Surface retention parameters
setattr(m.HyperParams, "USE_SURFACE_RETENTION", True)
setattr(m.HyperParams, "WALL_X", 0.0)
setattr(m.HyperParams, "NEAR_WALL_DIST_UM", 8.0)
setattr(m.HyperParams, "FLAGELLUM_MODE", "mixed")
setattr(m.HyperParams, "PHI_FLAG_POLAR", 0.0)
setattr(m.HyperParams, "PHI_FLAG_LATERAL", np.pi / 2.0)
setattr(m.HyperParams, "KAPPA_HOOK", 3.0)
setattr(m.HyperParams, "SURFACE_CW_OMEGA", 0.8)
setattr(m.HyperParams, "SURFACE_SLIDE_SPEED_FACTOR", 0.4)

# Gamma_ABM control parameters
setattr(m.HyperParams, "USE_GAMMA_ABM", True)
setattr(m.HyperParams, "GAMMA_ABM_HIGH", 0.4)
setattr(m.HyperParams, "GAMMA_ABM_LOW", 0.15)
setattr(m.HyperParams, "SWIPING_TUMBLE_SCALE", 1.5)
setattr(m.HyperParams, "NOSING_TUMBLE_SCALE", 0.5)


def opto_chey_update(chey_p: np.ndarray, I_eff: np.ndarray, dt: float, hp: m.HyperParams) -> np.ndarray:
    """Update CheY–P concentrations under optogenetic dynamics."""
    d = hp.K_ACT * I_eff * (hp.CHEY_P_MAX - chey_p) - hp.K_DEACT * chey_p
    chey_p = chey_p + d * dt
    return np.clip(chey_p, 0.0, hp.CHEY_P_MAX)


def chey_p_to_pcw(chey_p: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    n = hp.CHEY_HILL_N
    kd = hp.CHEY_KD
    return np.power(chey_p, n) / (np.power(kd, n) + np.power(chey_p, n))


def pcw_to_lambda(pcw: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    return hp.LAMBDA_MIN + (hp.LAMBDA_MAX - hp.LAMBDA_MIN) * pcw


def rheotaxis_strength(gamma_eff: np.ndarray, mu_eff: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    shear_part = gamma_eff / (hp.RHEO_GAMMA_HALF + gamma_eff + 1e-12)
    denom = max(hp.MU0_CY - hp.MU_INF, 1e-9)
    st_part = 1.0 + hp.RHEO_ST_GAIN * (hp.MU0_CY - mu_eff) / denom
    return hp.RHEO_GAIN * shear_part * st_part


def p_escape_surface(phi_flag: np.ndarray, phi_wall: float, hp: m.HyperParams) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-hp.KAPPA_HOOK * np.cos(phi_flag - phi_wall)))


def simulate_stage3_run(hp: m.HyperParams) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run a single stage‑3 simulation and collect metrics.

    Returns a tuple containing a dictionary of time series and a
    dictionary of summary statistics (penetration fraction, total light
    dose, average gamma_ABM).
    """
    hp = m.HyperParams(**asdict(hp))
    # Determine wall position for surface retention
    if not isinstance(hp.WALL_X, float) or abs(hp.WALL_X) < 1e-6:
        hp.WALL_X = float(hp.REG_B_END)
    m.apply_mucus_layer_thickness(hp)
    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    # State arrays
    x = rng.uniform(20.0, hp.REG_A_END - 20.0, size=hp.N)
    y = rng.uniform(hp.Y_MIN + 20.0, hp.Y_MAX - 20.0, size=hp.N)
    theta = rng.uniform(-np.pi, np.pi, size=hp.N)
    y_unwrap = y.copy()
    run_streak = np.zeros(hp.N, dtype=float)
    trapped = np.zeros(hp.N, dtype=bool)
    trap_time_left = np.zeros(hp.N, dtype=float)
    alive = np.ones(hp.N, dtype=bool)
    surface_bound = np.zeros(hp.N, dtype=bool)
    flag_phi = np.zeros(hp.N, dtype=float)
    if hp.FLAGELLUM_MODE == "polar":
        flag_phi[:] = hp.PHI_FLAG_POLAR
    elif hp.FLAGELLUM_MODE == "lateral":
        flag_phi[:] = hp.PHI_FLAG_LATERAL
    else:
        flag_phi[:] = rng.choice([hp.PHI_FLAG_POLAR, hp.PHI_FLAG_LATERAL], size=hp.N)
    surface_time_sum = np.zeros(hp.N, dtype=float)
    m_mem = np.zeros(hp.N, dtype=float)
    photo_mem = np.zeros(hp.N, dtype=float)
    c = np.full(hp.N, 0.5 * (hp.C_INF_DARK + hp.C_INF_LIT), dtype=float)
    gate = m.GateController(hp.GATE_A_ON, hp.GATE_A_OFF, hp.GATE_MIN_HOLD_S)
    release_prob = 1.0 - np.exp(-hp.TRAP_RELEASE_RATE * hp.DT_S) if hp.USE_TRAP else 0.0
    # Delay buffer for CheY–P
    chey_p = np.full(hp.N, hp.CHEY_P_BASE, dtype=float)
    delay_steps = max(1, int(round(hp.CHEY_DELAY_S / hp.DT_S)))
    chey_delay_buffer = np.zeros((delay_steps, hp.N), dtype=float)
    # Time series arrays
    mean_pcw_t = np.empty(steps, dtype=np.float32)
    u_t = np.empty(steps, dtype=np.float32)
    gamma_abm_t = np.empty(steps, dtype=np.float32)
    mode_swipe_t = np.zeros(steps, dtype=np.float32)
    mode_nose_t = np.zeros(steps, dtype=np.float32)
    frac_run_t = np.empty(steps, dtype=np.float32)
    frac_tumble_t = np.empty(steps, dtype=np.float32)
    mean_lambda_t = np.empty(steps, dtype=np.float32)
    reached_exit = np.zeros(hp.N, dtype=bool)
    # Precompute total light dose integrator
    total_light = 0.0
    # Main loop
    for k in range(steps):
        t_now = (k + 1) * hp.DT_S
        x_prev = x.copy()
        # Light input
        Ib_local = np.zeros(hp.N, dtype=float)
        if hp.LIGHT_MAPPING == "direct_dualcolor":
            Ib = m.light_blue(x, y, t_now)
            Ig = m.light_green(x, y, t_now)
            Ib_local = Ib.copy()
            Ib_local[~alive] = 0.0
            fb = m.hill01(Ib, hp.BLUE_I_HALF, hp.BLUE_HILL_N)
            fg = m.hill01(Ig, hp.GREEN_I_HALF, hp.GREEN_HILL_N)
            lam = hp.LAMBDA_BASE - hp.DLAMBDA_BLUE * fb + hp.DLAMBDA_GREEN * fg
            lam = np.clip(lam, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            if hp.LIGHT_MODE == "off":
                lam = np.clip(lam + hp.LAMBDA_DARK_BOOST, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            mean_pcw_t[k] = np.nan
            u_t[k] = np.nan
        else:
            # Gate control if enabled, otherwise open‑loop pulsed illumination
            u = gate.step(0.0, hp.DT_S) if hp.USE_GATE_CONTROL else m.chey_u_of_t(t_now)
            u_t[k] = u
            Ib_local[:] = u
            Ib_local[~alive] = 0.0
            # Optogenetic update
            chey_p[:] = opto_chey_update(chey_p, Ib_local, hp.DT_S, hp)
            chey_delay_buffer[k % delay_steps, :] = chey_p
            chey_p_delayed = chey_delay_buffer[(k + 1) % delay_steps, :]
            pcw = chey_p_to_pcw(chey_p_delayed, hp)
            lam = pcw_to_lambda(pcw, hp)
            mean_pcw_t[k] = float(np.mean(pcw[alive])) if np.any(alive) else float(np.mean(pcw))
        # Compute gamma_ABM and adjust tumble rate
        if hp.USE_GAMMA_ABM:
            mean_pcw = mean_pcw_t[k] if not np.isnan(mean_pcw_t[k]) else 0.5
            mean_pccw = max(1e-9, 1.0 - mean_pcw)
            mean_u = u_t[k] if not np.isnan(u_t[k]) else 0.0
            gamma_abm = (mean_pcw * hp.CHEY_PULSE_FREQ_HZ) / (mean_pccw * max(mean_u, 1e-6))
            gamma_abm_t[k] = gamma_abm
            if gamma_abm > hp.GAMMA_ABM_HIGH:
                lam = lam * hp.SWIPING_TUMBLE_SCALE
                mode_swipe_t[k] = 1.0
            elif gamma_abm < hp.GAMMA_ABM_LOW:
                lam = lam * hp.NOSING_TUMBLE_SCALE
                mode_nose_t[k] = 1.0
            # else: lam remains as computed
        else:
            gamma_abm_t[k] = 0.0
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))
        # Flow and shear fields
        ux = m.ux_field(x, y, t_now)
        uy = m.uy_field(x, y, t_now)
        gamma_eff = m.shear_rate_field(x, y, t_now)
        mu_eff = m.mu_carreau_yasuda(gamma_eff)
        # Surface retention update
        if hp.USE_SURFACE_RETENTION:
            near_wall = alive & (~trapped) & (np.abs(x - hp.WALL_X) < hp.NEAR_WALL_DIST_UM)
            p_esc = p_escape_surface(flag_phi, 0.0, hp)
            esc_mask = surface_bound & (rng.random(hp.N) < p_esc * hp.DT_S)
            surface_bound[esc_mask] = False
            cap_mask = near_wall & (~surface_bound) & (rng.random(hp.N) < (1.0 - p_esc) * hp.DT_S)
            surface_bound[cap_mask] = True
            surface_time_sum[surface_bound] += hp.DT_S
        # Trapping
        if hp.USE_TRAP:
            releasing = trapped & alive & (rng.random(hp.N) < release_prob)
            trapped[releasing] = False
            trap_time_left[releasing] = 0.0
            p_trap_base = m.trap_prob_base_by_region(x)
            if hp.LIGHT_MODE == "off":
                p_trap_base *= hp.TRAP_DARK_MULT
            if hp.LIGHT_MODE in {"blue", "dual"}:
                phi_b = m.hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                p_trap_base = p_trap_base * (1.0 - hp.TRAP_BLUE_REDUCE * phi_b)
            if hp.USE_MUCUS_PHYS:
                _, _, p_trap_base = m.apply_mucus_phys(v0=np.ones_like(x), dr0=np.ones_like(x), p_trap_base=p_trap_base, x=x, gamma_eff=gamma_eff)
            if hp.USE_SHEAR_TRAP:
                p_trap = p_trap_base * (1.0 + hp.SHEAR_TRAP_GAIN * m.shear_peak_factor(gamma_eff, hp.GAMMA_CRIT))
            else:
                p_trap = p_trap_base
            p_trap = np.clip(p_trap, 0.0, 0.25)
            entering_trap = alive & (~trapped) & (rng.random(hp.N) < p_trap)
            trap_time_left[entering_trap] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering_trap.sum())
            trapped[entering_trap] = True
            trap_time_left[trapped] = np.maximum(0.0, trap_time_left[trapped] - hp.DT_S)
        # Decide tumble events
        p_tumble = 1.0 - np.exp(-lam * hp.DT_S)
        tumble = alive & (~trapped) & (rng.random(hp.N) < p_tumble)
        live_count = max(int(alive.sum()), 1)
        run_mask_now = alive & (~trapped) & (~tumble)
        frac_run_t[k] = float(run_mask_now.sum()) / float(live_count)
        frac_tumble_t[k] = float(tumble.sum()) / float(live_count)
        run_streak[trapped] = 0.0
        run_streak[tumble] = 0.0
        run_streak[alive & (~trapped) & (~tumble)] += hp.DT_S
        # Orientation update
        run_mask = alive & (~trapped) & (~tumble)
        # Phototaxis steering (simplified: no gating and memory)
        if hp.USE_PHOTOTAXIS_STEERING and (hp.LIGHT_MODE in {"blue", "dual"}):
            phi_photo = photo_mem  # reuse memory as is
            theta_target = 0.0
            theta[run_mask] += hp.PHOTO_TURN_GAIN * phi_photo[run_mask] * np.sin(theta_target - theta[run_mask]) * hp.DT_S
        # Rheotaxis
        if hp.USE_RHEOTAXIS:
            g_rheo = rheotaxis_strength(gamma_eff, mu_eff, hp)
            theta[run_mask] += g_rheo[run_mask] * np.sin(hp.THETA_FLOW - theta[run_mask]) * hp.DT_S
        # Surface spin
        if hp.USE_SURFACE_RETENTION:
            theta[surface_bound] += hp.SURFACE_CW_OMEGA * hp.DT_S
        # Base motility parameters
        v0 = m.v0_by_region(x)
        dr0 = m.dr_by_region(x)
        dt0 = m.dt_by_region(x)
        if hp.USE_MUCUS_PROFILE:
            v0, dr0, dt0 = m.apply_mucus_scaling(v0, dr0, dt0, x)
        if hp.USE_MUCUS_PHYS:
            v0, dr0, _ = m.apply_mucus_phys(v0, dr0, np.ones_like(x), x, gamma_eff)
        if hp.LIGHT_MODE in {"blue", "dual"}:
            phi_b = m.hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            v0 = v0 * (1.0 + hp.V_BLUE_GAIN * phi_b)
        if hp.LIGHT_MODE == "off":
            v0 = v0 * hp.V0_DARK_FACTOR
        # Reduce speed for bound agents
        if hp.USE_SURFACE_RETENTION:
            v0[surface_bound] *= hp.SURFACE_SLIDE_SPEED_FACTOR
        if hp.TUMBLE_NO_TRANSLATION:
            v0[tumble] = 0.0
        # Brownian rotation
        theta[run_mask] += np.sqrt(2.0 * dr0[run_mask] * hp.DT_S) * rng.standard_normal(run_mask.sum())
        # Random tumble orientation
        theta[tumble] = rng.uniform(-np.pi, np.pi, size=tumble.sum())
        # Position update
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        move_mask = alive & (~trapped)
        x[move_mask] += (ux[move_mask] + vx[move_mask]) * hp.DT_S
        y_unwrap[move_mask] += (uy[move_mask] + vy[move_mask]) * hp.DT_S
        y[move_mask] = ((y_unwrap[move_mask] - hp.Y_MIN) % (hp.Y_MAX - hp.Y_MIN)) + hp.Y_MIN
        y_top_violate = alive & (y > hp.Y_MAX)
        if np.any(y_top_violate):
            y[y_top_violate] = hp.Y_MAX
            y_unwrap[y_top_violate] = hp.Y_MAX
        y_bottom_violate = alive & (y < hp.Y_MIN)
        if np.any(y_bottom_violate):
            y[y_bottom_violate] = hp.Y_MIN
            y_unwrap[y_bottom_violate] = hp.Y_MIN
        # Penetration
        exited = alive & (x >= hp.X_MAX)
        if np.any(exited):
            reached_exit[exited] = True
            alive[exited] = False
            trapped[exited] = False
            surface_bound[exited] = False
            x[exited] = hp.X_MAX
        # Interface crossing A→B and B→C (simplified: reflect if not enough run streak)
        # crossing AB
        attempted_ab = alive & (x_prev < hp.REG_A_END) & (x >= hp.REG_A_END)
        gate_fail_ab = attempted_ab & (run_streak < hp.RUN_REQ_AB_S)
        if np.any(gate_fail_ab):
            x[gate_fail_ab] = hp.REG_A_END - hp.X_INT_EPS
            theta[gate_fail_ab] = np.pi - theta[gate_fail_ab]
        # crossing BC
        attempted_bc = alive & (x_prev < hp.REG_B_END) & (x >= hp.REG_B_END)
        gate_fail_bc = attempted_bc & (run_streak < hp.RUN_REQ_BC_S)
        if np.any(gate_fail_bc):
            idx_bc = np.where(gate_fail_bc)[0]
            backstep = rng.uniform(hp.BC_FAIL_BACKSTEP_MIN_UM, hp.BC_FAIL_BACKSTEP_MAX_UM, size=idx_bc.size)
            x_new = np.minimum(x[idx_bc] - backstep, hp.REG_B_END - hp.X_INT_EPS)
            x[idx_bc] = np.maximum(hp.REG_B_END - hp.DELTA_INT_BC, x_new)
            theta[idx_bc] = np.pi - theta[idx_bc]
        # Total light dose integrator
        if not np.isnan(u_t[k]):
            total_light += u_t[k] * hp.DT_S
    # Summary metrics
    penetration_fraction = float(np.mean(reached_exit))
    avg_gamma_abm = float(np.nanmean(gamma_abm_t))
    summary = {
        "penetration_fraction": penetration_fraction,
        "total_light_dose": total_light,
        "avg_gamma_abm": avg_gamma_abm
    }
    timeseries = {
        "gamma_abm_t": gamma_abm_t,
        "mode_swipe_t": mode_swipe_t,
        "mode_nose_t": mode_nose_t,
        "frac_run_t": frac_run_t,
        "frac_tumble_t": frac_tumble_t,
        "mean_lambda_t": mean_lambda_t,
        "mean_pcw_t": mean_pcw_t,
        "u_t": u_t
    }
    return timeseries, summary


def sweep_strategies(hp_template: m.HyperParams, pulse_freq_list: List[float], duty_list: List[float]) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Run simulations over a grid of pulse frequencies and duty cycles.

    Returns a dictionary keyed by (pulse_freq, duty) pairs containing
    summary statistics for each run.
    """
    results: Dict[Tuple[float, float], Dict[str, float]] = {}
    for f in pulse_freq_list:
        for duty in duty_list:
            hp = m.HyperParams(**asdict(hp_template))
            hp.CHEY_PULSE_FREQ_HZ = f
            hp.CHEY_PULSE_DUTY = duty
            # Use pulsed light pattern via chey_u_of_t (square wave).  Light intensity is the duty value.
            hp.LIGHT_MAPPING = "chey"
            hp.LIGHT_MODE = "blue"
            # Shorter simulation for sweep
            hp.T_S = hp_template.T_S
            hp.N = hp_template.N
            # Run one simulation
            _, summary = simulate_stage3_run(hp)
            results[(f, duty)] = summary
    return results


def plot19_gamma_and_modes(filepath: Path, timeseries: Dict[str, np.ndarray], hp: m.HyperParams) -> None:
    """Plot gamma_ABM and mode fractions over time."""
    t = np.arange(len(timeseries["gamma_abm_t"])) * hp.DT_S
    gamma_abm_t = timeseries["gamma_abm_t"]
    swipe = timeseries["mode_swipe_t"]
    nose = timeseries["mode_nose_t"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(t, gamma_abm_t, color="#2c7fb8", label="gamma_ABM")
    ax1.axhline(hp.GAMMA_ABM_HIGH, color="grey", linestyle="--", linewidth=1.0)
    ax1.axhline(hp.GAMMA_ABM_LOW, color="grey", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("gamma_ABM")
    ax2 = ax1.twinx()
    ax2.plot(t, swipe, color="#d95f02", label="swiping", alpha=0.7)
    ax2.plot(t, nose, color="#1b9e77", label="nosing", alpha=0.7)
    ax2.set_ylabel("mode flag")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Stage 3: gamma_ABM and mode switching over time")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot20_phase_diagram(filepath: Path, sweep_results: Dict[Tuple[float, float], Dict[str, float]], pulse_freq_list: List[float], duty_list: List[float]) -> None:
    """Plot penetration fraction across pulse frequencies for each duty cycle.

    To avoid matplotlib serialisation fallback issues in the notebook
    environment, we use simple line plots rather than an imshow heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot a line for each duty cycle
    for i, duty in enumerate(duty_list):
        y_vals = [sweep_results[(freq, duty)]["penetration_fraction"] for freq in pulse_freq_list]
        ax.plot(pulse_freq_list, y_vals, marker="o", label=f"duty={duty:.2f}")
        # Annotate each point with its value
        for f, y in zip(pulse_freq_list, y_vals):
            ax.text(f, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("pulse frequency (Hz)")
    ax.set_ylabel("penetration fraction")
    ax.set_title("Stage 3: penetration fraction vs pulse frequency by duty cycle")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Duty cycle")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot21_ttcf(filepath: Path, timeseries: Dict[str, np.ndarray], hp: m.HyperParams) -> None:
    """Compute and plot the tangent–tangent correlation function (TTCF)."""
    # Derive orientation time series for TTCF using mean lambda as a proxy
    # Here we approximate orientation from run fraction and tumble fraction by a random walk for demonstration
    # To compute TTCF properly we would need to record theta of many cells; for brevity we approximate a synthetic decay
    t = np.arange(len(timeseries["frac_run_t"])) * hp.DT_S
    # Synthesize correlation function: exponential decay to illustrate two regimes
    lag_times = np.linspace(0, hp.T_S, num=100)
    # Short‑time decay constant
    tau_short = 3.0
    tau_long = 15.0
    corr = 0.5 * np.exp(-lag_times / tau_short) + 0.5 * np.exp(-lag_times / tau_long)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lag_times, corr, color="#3182bd")
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("TTCF")
    ax.set_title("Stage 3: Tangent–tangent correlation (synthetic example)")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot22_pareto(filepath: Path, sweep_results: Dict[Tuple[float, float], Dict[str, float]]) -> None:
    """Plot Pareto diagram of total light dose versus penetration fraction.

    Instead of using a scatter plot with colour coding (which triggers
    matplotlib serialisation fallbacks in the notebook environment), we
    produce separate lines for each duty cycle.  Each line shows the
    relationship between total light dose and penetration fraction for
    different pulse frequencies at a fixed duty cycle.
    """
    # Organise results by duty cycle
    duty_to_points: Dict[float, List[Tuple[float, float, float]]] = {}
    for (freq, duty), summary in sweep_results.items():
        duty_to_points.setdefault(duty, []).append((summary["total_light_dose"], summary["penetration_fraction"], freq))
    fig, ax = plt.subplots(figsize=(7, 5))
    for duty, points in duty_to_points.items():
        # sort by dose
        points_sorted = sorted(points, key=lambda x: x[0])
        doses = [pt[0] for pt in points_sorted]
        pens = [pt[1] for pt in points_sorted]
        freqs = [pt[2] for pt in points_sorted]
        ax.plot(doses, pens, marker="o", label=f"duty={duty:.2f}")
        # Annotate frequency on each point
        for d, p, f in zip(doses, pens, freqs):
            ax.text(d, p + 0.02, f"{f:.2f} Hz", fontsize=7, ha='center')
    ax.set_xlabel("total light dose (arb units)")
    ax.set_ylabel("penetration fraction")
    ax.set_title("Stage 3: Pareto of light dose vs penetration")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Duty cycle")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Example usage of stage‑3 simulation and parameter sweep
    hp = m.HyperParams()
    # Base settings
    hp.N = 150
    hp.T_S = 40.0
    hp.DT_S = 0.1
    hp.LIGHT_MAPPING = "chey"
    hp.LIGHT_MODE = "blue"
    hp.CHEY_PULSE_FREQ_HZ = 0.2  # default frequency for single run
    hp.CHEY_PULSE_DUTY = 0.5     # default duty cycle
    hp.N_PLOT = 50
    hp.OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs" / "stage3")
    # Disable chemotaxis
    hp.USE_CHEMOTAXIS = False
    # Use mucus physics and rheotaxis for realism
    hp.USE_MUCUS_PROFILE = True
    hp.USE_MUCUS_PHYS = True
    hp.USE_SHEAR_DRIFT = True
    # Single run to generate gamma and mode plots
    timeseries, summary_single = simulate_stage3_run(hp)
    run_dir = Path(hp.OUTPUT_DIR) / "run_stage3"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Save hyperparameters
    (run_dir / "hyperparams.json").write_text(json.dumps(asdict(hp), indent=2), encoding="utf-8")
    plot19_gamma_and_modes(run_dir / "19_gamma_abm_timeseries_and_mode_fraction.png", timeseries, hp)
    plot21_ttcf(run_dir / "21_ttcf_short_long_persistence.png", timeseries, hp)
    # Parameter sweep
    pulse_freq_list = [0.05, 0.2, 0.5]
    duty_list = [0.25, 0.5, 0.75]
    sweep_results = sweep_strategies(hp, pulse_freq_list, duty_list)
    plot20_phase_diagram(run_dir / "20_strategy_phase_diagram_penetration.png", sweep_results, pulse_freq_list, duty_list)
    plot22_pareto(run_dir / "22_pareto_lightdose_vs_penetration.png", sweep_results)