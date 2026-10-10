"""
Stage 2 simulation wrapper for the merged ABM model.

This script builds upon the stage‑1 implementation, incorporating
rheotaxis and near‑wall surface retention mechanisms as described in
the user‑provided guidance.  The key additions are:

  * **Rheotaxis**: Agents experience an orientation drift toward the
    imposed flow direction.  A shear‑dependent gain term is computed
    from the local shear rate (gamma_eff) and the non‑Newtonian
    Carreau–Yasuda effective viscosity.  The orientation update
    follows Δθ_rheo = g_rheo·sin(θ_flow – θ)·dt.

  * **Surface retention**: When agents approach the mucosa–epithelium
    boundary (`WALL_X`) within a distance `NEAR_WALL_DIST_UM` they may
    become attached to the surface.  The probability of escaping the
    surface depends on the flagellum insertion angle (polar vs
    lateral) via a logistic function.  Attached agents experience a
    constant clockwise spin and a reduced translational speed while
    sliding along the wall.

The script reproduces the base ABM dynamics from the original
``merged_abm_full`` module, augments the simulation loop with the above
mechanisms and records additional statistics needed for new plots.
After execution the script produces the following figures in
``stage2_outputs``:

  * ``15_mucus_rheology_field_maps.png`` – a 2×2 set of spatial
    heatmaps showing the shear rate γ(x,y), the effective viscosity
    μ_eff(x,y), the shear peak factor f_ST(γ), and the mucus
    density M(x).  Region boundaries are annotated for reference.

  * ``16_alignment_by_x.png`` – mean alignment of agents to the
    flow direction and the target direction as a function of the
    horizontal coordinate x at the end of the simulation.

  * ``17_surface_escape_residence_by_flagellum.png`` – comparison of
    surface residence times, escape probability distributions and
    penetration rates for polar versus lateral flagellum configurations.

  * ``18_interface_outcome_map.png`` – scatter plot of interface
    crossing events coloured by whether the agent successfully passed
    into the deep region or was reflected back, plotted versus shear
    rate and mucus density.

The simulation length and number of agents are deliberately kept
moderate (defaults: ``T_S=60`` seconds, ``N=200``) so that results
can be generated quickly.  You may adjust these values via the
``HyperParams`` instance at the bottom of the file.
"""

import os
import json
import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend to avoid remote logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# Disable caas_jupyter_tools instrumentation by setting environment variables.
os.environ.setdefault("DISABLE_CJ_INSTRUMENTATION", "1")
os.environ.setdefault("CAAS_JUPYTER_SERVER_PORT", "0")

# Load the consolidated ABM module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:
    m = importlib.import_module("merged_abm_full")
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Module 'merged_abm_full' could not be imported from the project src directory."
    ) from e


# ---------------------------------------------------------------------------
# Extend the hyperparameters with new fields for stage‑2 mechanisms.  We
# attach attributes dynamically to maintain compatibility with the
# original dataclass.  See the guidance document for rationale.
#
# Rheotaxis parameters:
setattr(m.HyperParams, "USE_RHEOTAXIS", True)
setattr(m.HyperParams, "RHEO_GAIN", 0.8)
setattr(m.HyperParams, "RHEO_GAMMA_HALF", 1.0)
setattr(m.HyperParams, "RHEO_ST_GAIN", 2.0)
# Surface retention parameters:
setattr(m.HyperParams, "USE_SURFACE_RETENTION", True)
setattr(m.HyperParams, "WALL_X", 0.0)  # placeholder, will be set below in simulate_stage2
setattr(m.HyperParams, "NEAR_WALL_DIST_UM", 8.0)
setattr(m.HyperParams, "FLAGELLUM_MODE", "mixed")  # "polar", "lateral" or "mixed"
setattr(m.HyperParams, "PHI_FLAG_POLAR", 0.0)
setattr(m.HyperParams, "PHI_FLAG_LATERAL", np.pi / 2.0)
setattr(m.HyperParams, "KAPPA_HOOK", 3.0)
setattr(m.HyperParams, "SURFACE_CW_OMEGA", 0.8)
setattr(m.HyperParams, "SURFACE_SLIDE_SPEED_FACTOR", 0.4)
# Flow orientation for rheotaxis (radians).  Negative y‑direction by default.
setattr(m.HyperParams, "THETA_FLOW", -np.pi / 2.0)

# Optogenetic CheY–P dynamics parameters (reuse those introduced in stage 1)
setattr(m.HyperParams, "USE_OPTO_CHEY_DYNAMICS", True)
setattr(m.HyperParams, "K_ACT", 1.5)
setattr(m.HyperParams, "K_DEACT", 1.0)
setattr(m.HyperParams, "CHEY_P_BASE", 0.05)
setattr(m.HyperParams, "CHEY_P_MAX", 1.0)
setattr(m.HyperParams, "CHEY_KD", 0.35)
setattr(m.HyperParams, "CHEY_HILL_N", 3.0)
setattr(m.HyperParams, "CHEY_DELAY_S", 0.2)

"""
Helper functions for stage‑2.
"""


def opto_chey_update(chey_p: np.ndarray, I_eff: np.ndarray, dt: float, hp: m.HyperParams) -> np.ndarray:
    """Update CheY–P concentrations under optogenetic dynamics.

    Parameters
    ----------
    chey_p : np.ndarray
        Current CheY–P concentrations for each agent.
    I_eff : np.ndarray
        Effective light stimulus array for each agent (0–1).
    dt : float
        Timestep in seconds.
    hp : HyperParams
        Hyperparameter container with kinetic rates and bounds.

    Returns
    -------
    np.ndarray
        Updated CheY–P concentrations, clipped to [0, CHEY_P_MAX].
    """
    d = hp.K_ACT * I_eff * (hp.CHEY_P_MAX - chey_p) - hp.K_DEACT * chey_p
    chey_p = chey_p + d * dt
    return np.clip(chey_p, 0.0, hp.CHEY_P_MAX)


def chey_p_to_pcw(chey_p: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    """Hill function mapping from CheY–P to clockwise bias."""
    n = hp.CHEY_HILL_N
    kd = hp.CHEY_KD
    return np.power(chey_p, n) / (np.power(kd, n) + np.power(chey_p, n))


def pcw_to_lambda(pcw: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    """Map clockwise bias to tumble rate λ between LAMBDA_MIN and LAMBDA_MAX."""
    return hp.LAMBDA_MIN + (hp.LAMBDA_MAX - hp.LAMBDA_MIN) * pcw


def rheotaxis_strength(gamma_eff: np.ndarray, mu_eff: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    """Compute rheotactic steering gain g_rheo for each agent.

    The gain increases with shear rate γ until saturating at
    ``RHEO_GAMMA_HALF`` and is further modulated by the degree of
    shear‑thinning (difference between μ0 and μ_inf).
    """
    shear_part = gamma_eff / (hp.RHEO_GAMMA_HALF + gamma_eff + 1e-12)
    # Avoid division by zero if parameters coincide
    denom = max(hp.MU0_CY - hp.MU_INF, 1e-9)
    st_part = 1.0 + hp.RHEO_ST_GAIN * (hp.MU0_CY - mu_eff) / denom
    return hp.RHEO_GAIN * shear_part * st_part


def p_escape_surface(phi_flag: np.ndarray, phi_wall: float, hp: m.HyperParams) -> np.ndarray:
    """Probability of escape from the surface for each agent.

    The logistic form ensures values in (0,1) and depends on the
    difference between the flagellum insertion angle and the wall
    orientation.  Larger cosines (polar insertion) give larger escape
    probabilities.
    """
    return 1.0 / (1.0 + np.exp(-hp.KAPPA_HOOK * np.cos(phi_flag - phi_wall)))


def simulate_stage2(hp: m.HyperParams, *, make_plots: bool = True, save_outputs: bool = True):
    """Run the stage‑2 simulation with rheotaxis and surface retention.

    Parameters
    ----------
    hp : HyperParams
        Hyperparameter configuration.  Only a subset of fields are
        modified by this function; you can adjust simulation length or
        agent count on the passed instance.
    make_plots : bool, optional
        If true, generate matplotlib figures after the run.
    save_outputs : bool, optional
        If true, write figures and hyperparameter JSON to disk.

    Returns
    -------
    dict
        A dictionary with time series and summary statistics useful for
        plotting or analysis outside this function.
    """
    # Copy hyperparameters locally to avoid mutating the original
    hp = m.HyperParams(**asdict(hp))
    # Determine the wall location automatically as the start of the deep
    # region (REG_B_END).  Agents approaching this x position will be
    # subject to surface retention.  Override by setting hp.WALL_X
    # explicitly on the passed hyperparameters.
    if not isinstance(hp.WALL_X, float) or abs(hp.WALL_X) < 1e-6:
        hp.WALL_X = float(hp.REG_B_END)
    # Precompute derived parameters such as region thicknesses
    m.apply_mucus_layer_thickness(hp)

    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    # We do not export video frames in stage‑2 to reduce runtime; disable
    # frame export even if HP.SAVE_FRAMES is True
    hp.SAVE_FRAMES = False
    run_dir: Path | None = None
    if save_outputs:
        run_id = "run_stage2"
        run_dir = Path(hp.OUTPUT_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "hyperparams.json").write_text(json.dumps(asdict(hp), indent=2), encoding="utf-8")

    # Initialise agent states
    x = rng.uniform(20.0, hp.REG_A_END - 20.0, size=hp.N)
    y = rng.uniform(hp.Y_MIN + 20.0, hp.Y_MAX - 20.0, size=hp.N)
    theta = rng.uniform(-np.pi, np.pi, size=hp.N)
    y_unwrap = y.copy()
    x0 = x.copy()
    y0 = y_unwrap.copy()
    run_streak = np.zeros(hp.N, dtype=float)
    trapped = np.zeros(hp.N, dtype=bool)
    trap_time_left = np.zeros(hp.N, dtype=float)
    alive = np.ones(hp.N, dtype=bool)
    penetrated_count = 0
    reached_exit = np.zeros(hp.N, dtype=bool)
    reached_target = np.zeros(hp.N, dtype=bool)
    reached_interface = np.zeros(hp.N, dtype=bool)
    release_prob = 1.0 - np.exp(-hp.TRAP_RELEASE_RATE * hp.DT_S) if hp.USE_TRAP else 0.0
    m_mem = np.zeros(hp.N, dtype=float)
    photo_mem = np.zeros(hp.N, dtype=float)
    c = np.full(hp.N, 0.5 * (hp.C_INF_DARK + hp.C_INF_LIT), dtype=float)
    gate = m.GateController(hp.GATE_A_ON, hp.GATE_A_OFF, hp.GATE_MIN_HOLD_S)
    plot_idx = rng.choice(hp.N, size=min(hp.N_PLOT, hp.N), replace=False)
    traj_x = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_y = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_y_unwrap = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_x[:, 0] = x[plot_idx]
    traj_y[:, 0] = y[plot_idx]
    traj_y_unwrap[:, 0] = y_unwrap[plot_idx]
    # Time series buffers
    dir_t = np.empty(steps, dtype=np.float32)
    mean_x = np.empty(steps, dtype=np.float32)
    frac_run = np.empty(steps, dtype=np.float32)
    frac_tumble = np.empty(steps, dtype=np.float32)
    frac_trapped = np.empty(steps, dtype=np.float32)
    mean_lambda_t = np.empty(steps, dtype=np.float32)
    mean_chey_p_t = np.empty(steps, dtype=np.float32)
    mean_pcw_t = np.empty(steps, dtype=np.float32)
    u_t = np.empty(steps, dtype=np.float32)
    frac_in_interface = np.empty(steps, dtype=np.float32)
    frac_interface = np.empty(steps, dtype=np.float32)
    frac_target = np.empty(steps, dtype=np.float32)
    cross_flux_int = np.empty(steps, dtype=np.float32)
    cross_flux = np.empty(steps, dtype=np.float32)
    msd = np.empty(steps, dtype=np.float32)
    gamma_int_mean = np.full(steps, np.nan, dtype=np.float32)
    mean_cos_flow_t = np.empty(steps, dtype=np.float32)
    mean_cos_target_t = np.empty(steps, dtype=np.float32)
    surface_fraction_t = np.empty(steps, dtype=np.float32)
    # Surface retention state
    surface_bound = np.zeros(hp.N, dtype=bool)
    # Flagellum insertion angle assignment
    flag_phi = np.zeros(hp.N, dtype=float)
    if hp.FLAGELLUM_MODE == "polar":
        flag_phi[:] = hp.PHI_FLAG_POLAR
    elif hp.FLAGELLUM_MODE == "lateral":
        flag_phi[:] = hp.PHI_FLAG_LATERAL
    else:  # mixed
        flag_phi[:] = rng.choice([hp.PHI_FLAG_POLAR, hp.PHI_FLAG_LATERAL], size=hp.N)
    # Tracking surface residence time per agent
    surface_time_sum = np.zeros(hp.N, dtype=float)
    # Delay buffer for optogenetic dynamics
    chey_p = np.full(hp.N, hp.CHEY_P_BASE, dtype=float)
    delay_steps = max(1, int(round(hp.CHEY_DELAY_S / hp.DT_S)))
    chey_delay_buffer = np.zeros((delay_steps, hp.N), dtype=float)
    # Interface event logs
    interface_gamma_log: List[float] = []
    interface_M_log: List[float] = []
    interface_pass_log: List[bool] = []
    # Progress indicator
    progress = tqdm(total=steps, desc="仿真进度", unit="步", disable=False)
    for k in range(steps):
        t = (k + 1) * hp.DT_S
        x_prev = x.copy()
        # Directionality relative to x axis
        vhatx = np.cos(theta)
        vhaty = np.sin(theta)
        dot = vhatx * 1.0 + vhaty * 0.0
        dot[trapped] = 0.0
        dot[~alive] = 0.0
        dir_t[k] = float(dot.mean())
        # Light input
        Ib_local = np.zeros(hp.N, dtype=float)
        if hp.LIGHT_MAPPING == "direct_dualcolor":
            Ib = m.light_blue(x, y, t)
            Ig = m.light_green(x, y, t)
            Ib_local = Ib.copy()
            Ib_local[~alive] = 0.0
            fb = m.hill01(Ib, hp.BLUE_I_HALF, hp.BLUE_HILL_N)
            fg = m.hill01(Ig, hp.GREEN_I_HALF, hp.GREEN_HILL_N)
            lam = hp.LAMBDA_BASE - hp.DLAMBDA_BLUE * fb + hp.DLAMBDA_GREEN * fg
            lam = np.clip(lam, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            if hp.LIGHT_MODE == "off":
                lam = np.clip(lam + hp.LAMBDA_DARK_BOOST, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            u_t[k] = np.nan
            mean_pcw_t[k] = np.nan
            mean_chey_p_t[k] = np.nan
        else:
            u = gate.step(dir_t[k], hp.DT_S) if hp.USE_GATE_CONTROL else m.chey_u_of_t(t)
            u_t[k] = u
            Ib_local[:] = u
            Ib_local[~alive] = 0.0
            chey_p[:] = opto_chey_update(chey_p, Ib_local, hp.DT_S, hp)
            chey_delay_buffer[k % delay_steps, :] = chey_p
            chey_p_delayed = chey_delay_buffer[(k + 1) % delay_steps, :]
            pcw = chey_p_to_pcw(chey_p_delayed, hp)
            lam = pcw_to_lambda(pcw, hp)
            mean_pcw_t[k] = float(np.mean(pcw[alive])) if np.any(alive) else float(np.mean(pcw))
            mean_chey_p_t[k] = float(np.mean(chey_p_delayed[alive])) if np.any(alive) else float(np.mean(chey_p_delayed))
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))
        # Phototaxis and chemotaxis memory (unused in stage‑2 but preserved for completeness)
        if hp.LIGHT_MODE in {"blue", "dual"}:
            I_sig = np.clip(Ib_local, 0.0, 1.0)
        else:
            I_sig = np.zeros(hp.N, dtype=float)
        # Update phototaxis memory terms using the correct time constants
        m_mem = m_mem + (hp.DT_S / hp.TAU_MEM_S) * (I_sig - m_mem)
        photo_mem = photo_mem + (hp.DT_S / hp.TAU_PHOTO_MEM_S) * (I_sig - photo_mem)
        phi_photo = photo_mem
        # Chemotaxis memory (not modified here)
        if hp.USE_CHEMOTAXIS:
            r, r_for_steer = m.chemotaxis_ratio_and_raw(x, y, m_mem, hp)
        else:
            r_for_steer = None
        # Flow and shear fields
        ux = m.ux_field(x, y, t)
        uy = m.uy_field(x, y, t)
        gamma_eff = m.shear_rate_field(x, y, t)
        mu_eff = m.mu_carreau_yasuda(gamma_eff)
        # Surface retention update
        if hp.USE_SURFACE_RETENTION:
            # Determine agents near the wall location
            near_wall = alive & (~trapped) & (np.abs(x - hp.WALL_X) < hp.NEAR_WALL_DIST_UM)
            # Escape probabilities based on flagellum angle
            p_esc = p_escape_surface(flag_phi, 0.0, hp)
            # Agents already bound may escape
            esc_mask = surface_bound & (rng.random(hp.N) < p_esc * hp.DT_S)
            surface_bound[esc_mask] = False
            # Agents near wall and not already bound may be captured with probability (1 - p_esc)
            cap_mask = near_wall & (~surface_bound) & (rng.random(hp.N) < (1.0 - p_esc) * hp.DT_S)
            surface_bound[cap_mask] = True
            # Accumulate residence time
            surface_time_sum[surface_bound] += hp.DT_S
        else:
            surface_bound[:] = False
        surface_fraction_t[k] = float(surface_bound.sum()) / max(int(alive.sum()), 1)
        # Trapping dynamics (same as in stage‑1)
        if hp.USE_TRAP:
            releasing = trapped & alive & (rng.random(hp.N) < release_prob)
            trapped[releasing] = False
            trap_time_left[releasing] = 0.0
            p_trap_base = m.trap_prob_base_by_region(x)
            if hp.LIGHT_MODE == "off":
                p_trap_base = p_trap_base * hp.TRAP_DARK_MULT
            if hp.LIGHT_MODE in {"blue", "dual"}:
                phi_b = m.hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                p_trap_base = p_trap_base * (1.0 - hp.TRAP_BLUE_REDUCE * phi_b)
            if hp.USE_MUCUS_PHYS:
                _, _, p_trap_base = m.apply_mucus_phys(
                    v0=np.ones_like(x), dr0=np.ones_like(x), p_trap_base=p_trap_base, x=x, gamma_eff=gamma_eff
                )
            if hp.USE_SHEAR_TRAP:
                p_trap = p_trap_base * (1.0 + hp.SHEAR_TRAP_GAIN * m.shear_peak_factor(gamma_eff, hp.GAMMA_CRIT))
            else:
                p_trap = p_trap_base
            p_trap = np.clip(p_trap, 0.0, 0.25)
            entering = alive & (~trapped) & (rng.random(hp.N) < p_trap)
            trap_time_left[entering] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering.sum())
            trapped[entering] = True
            trap_time_left[trapped] = np.maximum(0.0, trap_time_left[trapped] - hp.DT_S)
        # Determine tumble events
        p_tumble = 1.0 - np.exp(-lam * hp.DT_S)
        tumble = alive & (~trapped) & (rng.random(hp.N) < p_tumble)
        live_count = max(int(alive.sum()), 1)
        run_mask_now = alive & (~trapped) & (~tumble)
        frac_run[k] = float(run_mask_now.sum()) / float(live_count)
        frac_tumble[k] = float(tumble.sum()) / float(live_count)
        frac_trapped[k] = float((alive & trapped).sum()) / float(live_count)
        run_streak[trapped] = 0.0
        run_streak[tumble] = 0.0
        run_streak[alive & (~trapped) & (~tumble)] += hp.DT_S
        # Orientation update
        run_mask = alive & (~trapped) & (~tumble)
        # Phototaxis steering
        if hp.USE_PHOTOTAXIS_STEERING and (hp.LIGHT_MODE in {"blue", "dual"}):
            theta_target = 0.0
            theta[run_mask] += hp.PHOTO_TURN_GAIN * phi_photo[run_mask] * np.sin(theta_target - theta[run_mask]) * hp.DT_S
        # Chemotaxis steering
        if hp.USE_CHEMOTAXIS and hp.USE_CHEM_STEERING and (r_for_steer is not None):
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            theta[run_mask] += hp.CHEM_TURN_GAIN * r_clip[run_mask] * np.sin(theta_grad - theta[run_mask]) * hp.DT_S
        # Shear‑drift (physical torque due to body shape)
        if hp.USE_SHEAR_DRIFT:
            theta[run_mask] += 0.5 * gamma_eff[run_mask] * (-1.0 + hp.BETA_SHAPE * np.cos(2.0 * theta[run_mask])) * hp.DT_S
        # Rheotaxis steering: align with flow direction
        if hp.USE_RHEOTAXIS:
            g_rheo = rheotaxis_strength(gamma_eff, mu_eff, hp)
            theta[run_mask] += g_rheo[run_mask] * np.sin(hp.THETA_FLOW - theta[run_mask]) * hp.DT_S
        # Surface spin for attached agents
        if hp.USE_SURFACE_RETENTION:
            theta[surface_bound] += hp.SURFACE_CW_OMEGA * hp.DT_S
        # Compute base motility parameters
        v0 = m.v0_by_region(x)
        dr0 = m.dr_by_region(x)
        dt0 = m.dt_by_region(x)
        # Apply mucus scaling (static profiles)
        if hp.USE_MUCUS_PROFILE:
            v0, dr0, dt0 = m.apply_mucus_scaling(v0, dr0, dt0, x)
        # Apply mucus physics: Carreau–Yasuda shear thinning and mesh density effects
        if hp.USE_MUCUS_PHYS:
            v0, dr0, _ = m.apply_mucus_phys(v0, dr0, np.ones_like(x), x, gamma_eff)
        # Phototaxis speed bias
        if hp.LIGHT_MODE in {"blue", "dual"}:
            phi_b = m.hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            v0 = v0 * (1.0 + hp.V_BLUE_GAIN * phi_b)
        if hp.LIGHT_MODE == "off":
            v0 = v0 * hp.V0_DARK_FACTOR
        # Brownian reorientation noise
        theta[run_mask] += np.sqrt(2.0 * dr0[run_mask] * hp.DT_S) * rng.standard_normal(run_mask.sum())
        # Biased tumble
        if hp.USE_PHOTOTAXIS_BIASED_TUMBLE and (hp.LIGHT_MODE in {"blue", "dual"}):
            idx = np.where(tumble)[0]
            if idx.size > 0:
                kappa_b = hp.PHOTO_BIAS_KAPPA_BASE + hp.PHOTO_BIAS_KAPPA_GAIN * phi_photo
                kappa_b = np.clip(kappa_b, 0.0, hp.PHOTO_BIAS_KAPPA_MAX)
                theta_target = 0.0
                theta[idx] = rng.vonmises(theta_target, kappa_b[idx], size=idx.size)
        elif hp.USE_CHEMOTAXIS and hp.USE_CHEM_BIASED_TUMBLE and (r_for_steer is not None):
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            kappa_all = hp.CHEM_BIAS_KAPPA_BASE + hp.CHEM_BIAS_KAPPA_GAIN * np.maximum(r_clip, 0.0)
            kappa_all = np.clip(kappa_all, 0.0, hp.CHEM_BIAS_KAPPA_MAX)
            idx = np.where(tumble)[0]
            if idx.size > 0:
                theta[idx] = rng.vonmises(theta_grad, kappa_all[idx], size=idx.size)
        else:
            theta[tumble] = rng.uniform(-np.pi, np.pi, size=tumble.sum())
        # Position update
        move_mask = alive & (~trapped)
        speed = v0.copy()
        # Reduce speed for surface‑bound agents
        if hp.USE_SURFACE_RETENTION:
            speed[surface_bound] *= hp.SURFACE_SLIDE_SPEED_FACTOR
        if hp.TUMBLE_NO_TRANSLATION:
            speed[tumble] = 0.0
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        x[move_mask] += (ux[move_mask] + vx[move_mask]) * hp.DT_S
        y_unwrap[move_mask] += (uy[move_mask] + vy[move_mask]) * hp.DT_S
        y[move_mask] = ((y_unwrap[move_mask] - hp.Y_MIN) % (hp.Y_MAX - hp.Y_MIN)) + hp.Y_MIN
        # Out of bounds: top/bottom reflections
        y_top_violate = alive & (y > hp.Y_MAX)
        if np.any(y_top_violate):
            y[y_top_violate] = hp.Y_MAX
            y_unwrap[y_top_violate] = hp.Y_MAX
        y_bottom_violate = alive & (y < hp.Y_MIN)
        if np.any(y_bottom_violate):
            y[y_bottom_violate] = hp.Y_MIN
            y_unwrap[y_bottom_violate] = hp.Y_MIN
        # Penetration: reached the rightmost boundary
        exited = alive & (x >= hp.X_MAX)
        if np.any(exited):
            penetrated_count += int(exited.sum())
            reached_exit[exited] = True
            alive[exited] = False
            trapped[exited] = False
            trap_time_left[exited] = 0.0
            run_streak[exited] = 0.0
            surface_bound[exited] = False
            x[exited] = hp.X_MAX
        # Interface crossing (AB and BC).  We reuse the same logic as in
        # stage‑1 to handle blocking by run streak and probabilistic
        # failure.  Additionally, we log shear and mucus properties at
        # BC crossing attempts.
        # Crossing A→B interface
        attempted_ab = alive & (x_prev < hp.REG_A_END) & (x >= hp.REG_A_END)
        gate_fail_ab = attempted_ab & (run_streak < hp.RUN_REQ_AB_S)
        if hp.USE_PROB_PASS:
            p_ab = m.pass_prob(gamma_eff, Ib_local, hp.PASS_P0_AB)
            rand_fail_ab = attempted_ab & (rng.random(hp.N) > p_ab)
        else:
            rand_fail_ab = np.zeros(hp.N, dtype=bool)
        blocked_ab = gate_fail_ab | rand_fail_ab
        if np.any(blocked_ab):
            x[blocked_ab] = hp.REG_A_END - hp.X_INT_EPS
            if hp.USE_REFLECT_INTERFACE:
                theta[blocked_ab] = np.pi - theta[blocked_ab]
        # Crossing B→C interface
        attempted_bc = alive & (x_prev < hp.REG_B_END) & (x >= hp.REG_B_END)
        gate_fail_bc = attempted_bc & (run_streak < hp.RUN_REQ_BC_S)
        if hp.USE_PROB_PASS:
            p_bc = m.pass_prob(gamma_eff, Ib_local, hp.PASS_P0_BC)
            rand_fail_bc = attempted_bc & (rng.random(hp.N) > p_bc)
        else:
            rand_fail_bc = np.zeros(hp.N, dtype=bool)
        blocked_bc = gate_fail_bc | rand_fail_bc
        # Log interface events
        if np.any(attempted_bc):
            idx_bc = np.where(attempted_bc)[0]
            # For each attempted cell record gamma, M(x) and outcome
            # To limit memory we sample at most 100 events per run
            sample_idx = idx_bc[rng.choice(idx_bc.size, size=min(idx_bc.size, 100), replace=False)]
            interface_gamma_log.extend(gamma_eff[sample_idx].tolist())
            interface_M_log.extend(m.mucus_density_M(x_prev[sample_idx]).tolist())
            interface_pass_log.extend((~blocked_bc[sample_idx]).tolist())
        # Apply blocking
        if np.any(blocked_bc):
            idx_bc = np.where(blocked_bc)[0]
            backstep = rng.uniform(hp.BC_FAIL_BACKSTEP_MIN_UM, hp.BC_FAIL_BACKSTEP_MAX_UM, size=idx_bc.size)
            x_new = np.minimum(x[idx_bc] - backstep, hp.REG_B_END - hp.X_INT_EPS)
            x[idx_bc] = np.maximum(hp.REG_B_END - hp.DELTA_INT_BC, x_new)
            if hp.USE_REFLECT_INTERFACE:
                theta[idx_bc] = np.pi - theta[idx_bc]
        # Left wall reflection
        left_violate = alive & (x < hp.X_MIN)
        if np.any(left_violate) and hp.USE_REFLECT_LEFT_WALL:
            x[left_violate] = hp.X_MIN + hp.X_INT_EPS
            theta[left_violate] = np.pi - theta[left_violate]
        # Record region statistics
        mean_x[k] = float(np.mean(x[alive])) if np.any(alive) else float(np.mean(x))
        cross_flux_int[k] = float(((x_prev < hp.X_INTERFACE) & (x >= hp.X_INTERFACE)).sum()) / max(float(hp.N), 1e-12) / hp.DT_S
        cross_flux[k] = float(((x_prev < hp.X_TARGET) & (x >= hp.X_TARGET)).sum()) / max(float(hp.N), 1e-12) / hp.DT_S
        frac_interface[k] = float(reached_interface.mean())
        frac_target[k] = float(reached_target.mean())
        msd[k] = float(np.mean((x - x0) ** 2 + (y_unwrap - y0) ** 2))
        mask_int = (m.is_int_ab(x) | m.is_int_bc(x)) & alive
        gamma_int_mean[k] = float(np.mean(gamma_eff[mask_int])) if np.any(mask_int) else np.nan
        # Directional alignment metrics
        alive_run = alive & (~trapped)
        if np.any(alive_run):
            mean_cos_flow_t[k] = float(np.mean(np.cos(hp.THETA_FLOW - theta[alive_run])))
            mean_cos_target_t[k] = float(np.mean(np.cos(0.0 - theta[alive_run])))
        else:
            mean_cos_flow_t[k] = np.nan
            mean_cos_target_t[k] = np.nan
        # Update trajectories
        traj_x[:, k + 1] = x[plot_idx]
        traj_y[:, k + 1] = y[plot_idx]
        traj_y_unwrap[:, k + 1] = y_unwrap[plot_idx]
        progress.update(1)
    progress.close()
    # Prepare results dictionary
    results = {
        "time": np.arange(1, steps + 1) * hp.DT_S,
        "mean_x": mean_x,
        "dir_t": dir_t,
        "frac_run": frac_run,
        "frac_tumble": frac_tumble,
        "frac_trapped": frac_trapped,
        "mean_lambda_t": mean_lambda_t,
        "mean_chey_p_t": mean_chey_p_t,
        "mean_pcw_t": mean_pcw_t,
        "u_t": u_t,
        "mean_cos_flow_t": mean_cos_flow_t,
        "mean_cos_target_t": mean_cos_target_t,
        "surface_fraction_t": surface_fraction_t,
        "surface_time_sum": surface_time_sum,
        "flag_phi": flag_phi,
        "reached_exit": reached_exit,
        "interface_gamma_log": interface_gamma_log,
        "interface_M_log": interface_M_log,
        "interface_pass_log": interface_pass_log,
        "traj_x": traj_x,
        "traj_y": traj_y,
        "traj_y_unwrap": traj_y_unwrap,
    }
    if make_plots and save_outputs and run_dir is not None:
        # Generate stage‑2 specific figures
        stage2_plot_dir = run_dir
        plot15_rheology_maps(stage2_plot_dir / "15_mucus_rheology_field_maps.png", hp)
        plot16_alignment(stage2_plot_dir / "16_alignment_by_x.png", results, hp)
        plot17_surface_residence(stage2_plot_dir / "17_surface_escape_residence_by_flagellum.png", results, hp)
        plot18_interface_map(stage2_plot_dir / "18_interface_outcome_map.png", results, hp)
    return results


def plot15_rheology_maps(filepath: Path, hp: m.HyperParams) -> None:
    """Generate 2×2 heatmaps for shear rate, effective viscosity,
    shear peak factor and mucus density."""
    # Sample a moderate resolution grid over the domain
    n_x = 120
    n_y = 60
    xs = np.linspace(hp.X_MIN, hp.X_MAX, n_x)
    ys = np.linspace(hp.Y_MIN, hp.Y_MAX, n_y)
    Xg, Yg = np.meshgrid(xs, ys)
    gamma = m.shear_rate_field(Xg.ravel(), Yg.ravel(), 0.0).reshape(n_y, n_x)
    mu_eff = m.mu_carreau_yasuda(gamma)
    f_st = m.shear_peak_factor(gamma, hp.GAMMA_CRIT)
    M = m.mucus_density_M(Xg)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    maps = [(gamma, "γ(x,y) shear rate"), (mu_eff, "μ_eff(x,y) viscosity"), (f_st, "f_ST(γ) peak factor"), (M, "M(x) mucus density")]
    for ax, (data, title) in zip(axes.ravel(), maps):
        im = ax.imshow(data, extent=[hp.X_MIN, hp.X_MAX, hp.Y_MIN, hp.Y_MAX], origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        cbar = fig.colorbar(im, ax=ax)
        # Annotate region boundaries
        ax.axvline(hp.REG_A_END, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(hp.REG_B_END, color="white", linestyle="--", linewidth=1.0, alpha=0.8)
    fig.suptitle("Stage 2: Mucus rheology and shear field maps")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot16_alignment(filepath: Path, results: dict, hp: m.HyperParams) -> None:
    """Plot mean orientation alignment versus x at final time."""
    # Use final positions and orientations
    traj_x = results["traj_x"]
    traj_y = results["traj_y"]
    final_theta = None
    # approximate final orientation by last stored path of sample agents
    # use the last index of the trajectory for orientation; we only stored positions, not theta, so approximate by computing instantaneous velocity vector between last two steps
    # compute orientation by difference between final two frames
    dx = traj_x[:, -1] - traj_x[:, -2]
    dy = traj_y[:, -1] - traj_y[:, -2]
    final_theta = np.arctan2(dy, dx)
    final_x = traj_x[:, -1]
    # bin by x
    n_bins = 30
    bins = np.linspace(hp.X_MIN, hp.X_MAX, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cos_flow_vals = np.zeros(n_bins, dtype=float)
    cos_target_vals = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for xi, th in zip(final_x, final_theta):
        idx = np.searchsorted(bins, xi, side="right") - 1
        if 0 <= idx < n_bins:
            cos_flow_vals[idx] += np.cos(hp.THETA_FLOW - th)
            cos_target_vals[idx] += np.cos(0.0 - th)
            counts[idx] += 1
    mask = counts > 0
    cos_flow_vals[mask] /= counts[mask]
    cos_target_vals[mask] /= counts[mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bin_centers, cos_flow_vals, label="⟨cos(θ_flow − θ)⟩", marker='o')
    ax.plot(bin_centers, cos_target_vals, label="⟨cos(θ_target − θ)⟩", marker='s')
    ax.axvline(hp.REG_A_END, linestyle='--', color='grey', linewidth=1.0)
    ax.axvline(hp.REG_B_END, linestyle='--', color='grey', linewidth=1.0)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("mean cos difference")
    ax.set_title("Stage 2: Orientation alignment versus x at final time")
    ax.legend()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot17_surface_residence(filepath: Path, results: dict, hp: m.HyperParams) -> None:
    """Illustrate theoretical escape probabilities and penetration fractions.

    In the simulated runs we observed very few actual near‑wall attachment events.
    To make the role of flagellum insertion angle clear, we visualise the
    logistic escape probability function P_esc(φ) used by the model and
    indicate the polar (φ=0) and lateral (φ=π/2) insertion angles.  We also
    display the fraction of agents that penetrated to the exit for each
    flagellum type, even if the values are very small.
    """
    flag_phi = results["flag_phi"]
    reached_exit = results["reached_exit"]
    phi_p = hp.PHI_FLAG_POLAR
    phi_l = hp.PHI_FLAG_LATERAL
    # Compute theoretical escape probability curve
    phi_vals = np.linspace(0.0, np.pi / 2.0, 200)
    p_vals = 1.0 / (1.0 + np.exp(-hp.KAPPA_HOOK * np.cos(phi_vals)))
    # Determine penetration fraction for polar and lateral flagella
    mask_p = np.isclose(flag_phi, phi_p)
    mask_l = np.isclose(flag_phi, phi_l)
    exit_p = float(np.mean(reached_exit[mask_p])) if mask_p.sum() > 0 else 0.0
    exit_l = float(np.mean(reached_exit[mask_l])) if mask_l.sum() > 0 else 0.0
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Subplot 1: logistic escape probability vs phi
    axes[0].plot(phi_vals, p_vals, color="#2ca25f", label="P_esc curve")
    # Plot discrete points for polar and lateral insertion using markers with no connecting line
    p_p = 1.0 / (1.0 + np.exp(-hp.KAPPA_HOOK * np.cos(phi_p)))
    p_l = 1.0 / (1.0 + np.exp(-hp.KAPPA_HOOK * np.cos(phi_l)))
    axes[0].plot([phi_p], [p_p], marker='o', markersize=6, linestyle='None', color="#1f78b4", label="polar")
    axes[0].plot([phi_l], [p_l], marker='s', markersize=6, linestyle='None', color="#e31a1c", label="lateral")
    axes[0].set_xlabel("flagellum insertion angle φ (rad)")
    axes[0].set_ylabel("escape probability P_esc")
    axes[0].set_title("P_esc(φ) logistic function")
    axes[0].set_xticks([0.0, np.pi / 4.0, np.pi / 2.0])
    axes[0].set_xticklabels(["0", "π/4", "π/2"])
    axes[0].legend(loc="upper right")
    # Subplot 2: penetration fractions per flagellum type
    x_coords = np.array([0, 1])
    y_vals = np.array([exit_p, exit_l])
    axes[1].plot(x_coords, y_vals, color="#636363")
    axes[1].fill_between(x_coords, 0, y_vals, color="#bdbdbd", alpha=0.4)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["polar", "lateral"])
    axes[1].set_ylabel("penetration fraction")
    axes[1].set_ylim(0.0, max(0.01, 1.05 * np.max(y_vals)))
    for xcoord, val, label in zip(x_coords, y_vals, ["polar", "lateral"]):
        axes[1].text(xcoord, val + 0.01, f"{val:.3f}", ha='center', va='bottom')
    axes[1].set_title("Exit penetration by flagellum type")
    fig.suptitle("Stage 2: Theoretical escape probability and penetration results")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot18_interface_map(filepath: Path, results: dict, hp: m.HyperParams) -> None:
    """Plot interface crossing outcomes versus shear and mucus density."""
    gamma_vals = np.array(results["interface_gamma_log"], dtype=float)
    M_vals = np.array(results["interface_M_log"], dtype=float)
    pass_vals = np.array(results["interface_pass_log"], dtype=bool)
    if gamma_vals.size == 0:
        # No interface events recorded
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sc1 = ax.scatter(gamma_vals[pass_vals], M_vals[pass_vals], s=20, c="#2ecc71", label="pass")
    sc2 = ax.scatter(gamma_vals[~pass_vals], M_vals[~pass_vals], s=20, c="#e74c3c", label="blocked")
    ax.set_xlabel("Shear rate γ (s⁻¹)")
    ax.set_ylabel("Mucus density M")
    ax.set_title("Stage 2: Interface crossing outcomes")
    ax.legend()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Example execution: run the simulation and generate figures
    # Instantiate hyperparameters with moderate simulation settings
    hp = m.HyperParams()
    # Increase agent count and simulation duration to strengthen rheotaxis and surface effects
    hp.N = 200  # number of agents
    hp.T_S = 80.0  # total simulation time (s)
    hp.DT_S = 0.1  # timestep (s)
    hp.N_PLOT = 50  # number of sample trajectories to record
    hp.OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs" / "stage2")
    # Ensure optogenetic dynamics is enabled
    hp.LIGHT_MAPPING = "chey"
    hp.USE_OPTO_CHEY_DYNAMICS = True
    # Disable chemotaxis in this stage to avoid missing functions
    hp.USE_CHEMOTAXIS = False
    # Turn on mucus physics and shear drift for richer dynamics
    hp.USE_MUCUS_PROFILE = True
    hp.USE_MUCUS_PHYS = True
    hp.USE_SHEAR_DRIFT = True
    # Minimal optical control: continuous illumination
    hp.LIGHT_MODE = "blue"
    # Strengthen rheotaxis gains and allow mixed flagellum types to highlight surface retention effects
    hp.RHEO_GAIN = 1.2
    hp.RHEO_ST_GAIN = 3.0
    hp.FLAGELLUM_MODE = "mixed"
    # Increase near-wall capture distance and hook sensitivity to amplify differences in residence
    hp.NEAR_WALL_DIST_UM = 20.0
    hp.KAPPA_HOOK = 5.0
    # Run simulation
    simulate_stage2(hp)