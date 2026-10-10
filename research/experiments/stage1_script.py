"""
Stage 1 simulation wrapper for the merged ABM model.

This script imports the consolidated ABM implementation from `src/merged_abm_full.py`
and installs modifications that implement optogenetic CheY–P dynamics and
additional diagnostic plots. The core simulation loop is reimplemented here
to avoid editing the original code. When executed, the script runs a
shortened simulation (60 s with 200 agents) and outputs figures into `research/experiments/outputs/stage1`.
You can adjust the hyperparameters at the end of the file.

Run this script from the repository root. It loads the model from `src/`.
"""

import json
from pathlib import Path
from dataclasses import asdict
import sys
import numpy as np
import os
import matplotlib
# Disable caas_jupyter_tools instrumentation and use non-interactive backend
os.environ.setdefault("DISABLE_CJ_INSTRUMENTATION", "1")
os.environ.setdefault("CAAS_JUPYTER_SERVER_PORT", "0")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the consolidated ABM module.
import importlib
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

try:
    m = importlib.import_module("merged_abm_full")
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Module 'merged_abm_full' could not be imported from the project src directory."
    ) from e

# Extend the HyperParams dataclass with optogenetic parameters. We add these
# attributes dynamically to maintain compatibility with the original class.
setattr(m.HyperParams, "USE_OPTO_CHEY_DYNAMICS", True)
setattr(m.HyperParams, "K_ACT", 1.5)
setattr(m.HyperParams, "K_DEACT", 1.0)
setattr(m.HyperParams, "CHEY_P_BASE", 0.05)
setattr(m.HyperParams, "CHEY_P_MAX", 1.0)
setattr(m.HyperParams, "CHEY_KD", 0.35)
setattr(m.HyperParams, "CHEY_HILL_N", 3.0)
setattr(m.HyperParams, "CHEY_DELAY_S", 0.2)


# Helper functions for the optogenetic cascade
def opto_chey_update(chey_p: np.ndarray, I_eff: np.ndarray, dt: float, hp: m.HyperParams) -> np.ndarray:
    d = hp.K_ACT * I_eff * (hp.CHEY_P_MAX - chey_p) - hp.K_DEACT * chey_p
    chey_p = chey_p + d * dt
    return np.clip(chey_p, 0.0, hp.CHEY_P_MAX)


def chey_p_to_pcw(chey_p: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    n = hp.CHEY_HILL_N
    kd = hp.CHEY_KD
    return np.power(chey_p, n) / (np.power(kd, n) + np.power(chey_p, n))


def pcw_to_lambda(pcw: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    return hp.LAMBDA_MIN + (hp.LAMBDA_MAX - hp.LAMBDA_MIN) * pcw


def simulate_stage1(hp: m.HyperParams, *, make_plots: bool = True, save_outputs: bool = True):
    """Reimplementation of the simulation loop with optogenetic CheY–P dynamics."""
    # Copy hyperparameters locally and perform derived parameter adjustments
    hp = m.HyperParams(**asdict(hp))
    m.apply_mucus_layer_thickness(hp)

    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    n_frames = m._frame_count(hp)
    frame_steps = max(1, int(round(hp.FRAME_STEP_S / hp.DT_S)))
    run_dir: Path | None = None
    plot_dirs: dict[str, Path] = {}
    frame_manifest: list[dict[str, str]] = []
    window_steps = max(1, int(hp.WINDOW_S / hp.DT_S))
    if save_outputs:
        run_id = "run_stage1"
        run_dir = Path(hp.OUTPUT_DIR) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if hp.SAVE_FRAMES:
            plots_root = run_dir / hp.PLOT_FRAMES_SUBDIR
            for plot_id, _ in m._get_plot_video_specs(hp):
                plot_dirs[plot_id] = plots_root / plot_id
                plot_dirs[plot_id].mkdir(parents=True, exist_ok=True)
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
    reached_interface = np.zeros(hp.N, dtype=bool)
    reached_target = np.zeros(hp.N, dtype=bool)
    reached_exit = np.zeros(hp.N, dtype=bool)
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
    frac_interface = np.empty(steps, dtype=np.float32)
    cross_flux_int = np.empty(steps, dtype=np.float32)
    frac_target = np.empty(steps, dtype=np.float32)
    cross_flux = np.empty(steps, dtype=np.float32)
    msd = np.empty(steps, dtype=np.float32)
    u_t = np.empty(steps, dtype=np.float32)
    frac_run = np.empty(steps, dtype=np.float32)
    frac_tumble = np.empty(steps, dtype=np.float32)
    frac_trapped = np.empty(steps, dtype=np.float32)
    mean_lambda_t = np.empty(steps, dtype=np.float32)
    mean_c_t = np.empty(steps, dtype=np.float32)
    mean_pcw_t = np.empty(steps, dtype=np.float32)
    mean_chey_p_t = np.empty(steps, dtype=np.float32)
    trap_events = np.zeros(steps, dtype=np.int32)
    trap_events_int = np.zeros(steps, dtype=np.int32)
    frac_in_interface = np.zeros(steps, dtype=np.float32)
    gamma_int_mean = np.full(steps, np.nan, dtype=np.float32)
    x_edges_xt = np.linspace(hp.X_MIN, hp.X_MAX, hp.XT_BINS_X + 1)
    xt_density = np.zeros((steps, hp.XT_BINS_X), dtype=np.float32)
    x_edges_xy = np.linspace(hp.X_MIN, hp.X_MAX, hp.X_BINS_HEATMAP + 1)
    y_edges_xy = np.linspace(hp.Y_MIN, hp.Y_MAX, hp.Y_BINS_HEATMAP + 1)
    occupancy_xy = np.zeros((hp.Y_BINS_HEATMAP, hp.X_BINS_HEATMAP), dtype=np.float32)
    t_first_interface = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_target = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_exit = np.full(hp.N, np.nan, dtype=np.float32)
    x_prev = x.copy()
    d_hat = np.array([1.0, 0.0], dtype=float)
    use_frame_export = bool(save_outputs and hp.SAVE_FRAMES and plot_dirs)
    progress = tqdm(
        total=n_frames if use_frame_export else steps,
        desc="仿真进度" if use_frame_export else "仿真进度",
        unit="帧" if use_frame_export else "步",
        disable=False,
    )
    chey_p = np.full(hp.N, hp.CHEY_P_BASE, dtype=float)
    delay_steps = max(1, int(round(hp.CHEY_DELAY_S / hp.DT_S)))
    chey_delay_buffer = np.zeros((delay_steps, hp.N), dtype=float)
    for k in range(steps):
        t = (k + 1) * hp.DT_S
        x_prev[:] = x
        # Directionality
        vhatx = np.cos(theta)
        vhaty = np.sin(theta)
        dot = vhatx * d_hat[0] + vhaty * d_hat[1]
        dot[trapped] = 0.0
        dot[~alive] = 0.0
        A = float(dot.mean())
        dir_t[k] = A
        # Light input and lambda
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
        else:
            u = gate.step(A, hp.DT_S) if hp.USE_GATE_CONTROL else m.chey_u_of_t(t)
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
            mean_c_t[k] = mean_chey_p_t[k]
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))
        # Phototaxis memory
        if hp.LIGHT_MODE in {"blue", "dual"}:
            I_sig = np.clip(Ib_local, 0.0, 1.0)
        else:
            I_sig = np.zeros(hp.N, dtype=float)
        phi_I = m.hill01(np.clip(I_sig, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
        if hp.USE_PHOTO_MEMORY:
            alpha_p = np.exp(-hp.DT_S / max(hp.TAU_PHOTO_MEM_S, 1e-12))
            photo_mem[:] = I_sig + (photo_mem - I_sig) * alpha_p
            r_photo = I_sig - photo_mem
            r_photo = np.clip(r_photo, -hp.PHOTO_R_CLIP, hp.PHOTO_R_CLIP)
        else:
            r_photo = I_sig
        phi_d = np.maximum(r_photo, 0.0) / max(hp.PHOTO_R_CLIP, 1e-12)
        phi_d = np.clip(phi_d, 0.0, 1.0)
        phi_photo = np.clip(phi_d + hp.PHOTO_ABS_WEIGHT * phi_I, 0.0, 1.0)
        # Chemotaxis modulation
        if hp.USE_CHEMOTAXIS:
            C = m.nutrient_C(x, y)
            S = m.chem_signal(C)
            m_mem[:] = m.chem_update_memory(m_mem, S, hp.DT_S)
            r = S - m_mem
            lam = m.chem_modulate_lambda(lam, r)
            r_for_steer = r
        else:
            r_for_steer = None
        # Flow and shear
        ux = m.ux_field(x, y, t)
        uy = m.uy_field(x, y, t)
        gamma_eff = m.shear_rate_field(x, y, t)
        # Trapping dynamics
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
            trap_events[k] = int(entering.sum())
            in_int_layer = m.is_int_ab(x) | m.is_int_bc(x)
            trap_events_int[k] = int((entering & in_int_layer).sum())
            trap_time_left[entering] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering.sum())
            trapped[entering] = True
            trap_time_left[trapped] = np.maximum(0.0, trap_time_left[trapped] - hp.DT_S)
        # Decide tumble events
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
        if hp.USE_PHOTOTAXIS_STEERING and (hp.LIGHT_MODE in {"blue", "dual"}):
            theta_target = 0.0
            theta[run_mask] += hp.PHOTO_TURN_GAIN * phi_photo[run_mask] * np.sin(theta_target - theta[run_mask]) * hp.DT_S
        if hp.USE_CHEMOTAXIS and hp.USE_CHEM_STEERING and (r_for_steer is not None):
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            theta[run_mask] += hp.CHEM_TURN_GAIN * r_clip[run_mask] * np.sin(theta_grad - theta[run_mask]) * hp.DT_S
        if hp.USE_SHEAR_DRIFT:
            theta[run_mask] += 0.5 * gamma_eff[run_mask] * (-1.0 + hp.BETA_SHAPE * np.cos(2.0 * theta[run_mask])) * hp.DT_S
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
        if hp.TUMBLE_NO_TRANSLATION:
            speed[tumble] = 0.0
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        x[move_mask] += (ux[move_mask] + vx[move_mask]) * hp.DT_S
        y_unwrap[move_mask] += (uy[move_mask] + vy[move_mask]) * hp.DT_S
        y[move_mask] = y_unwrap[move_mask]
        if hp.USE_Y_PERIODIC:
            span = hp.Y_MAX - hp.Y_MIN
            y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN
        if hp.USE_TRANSLATIONAL_DIFFUSION:
            x[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())
            y_unwrap[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())
            y[move_mask] = y_unwrap[move_mask]
            if hp.USE_Y_PERIODIC:
                span = hp.Y_MAX - hp.Y_MIN
                y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN
        newly_interface = (~reached_interface) & (x >= hp.X_INTERFACE)
        newly_target = (~reached_target) & (x >= hp.X_TARGET)
        t_first_interface[newly_interface] = t
        t_first_target[newly_target] = t
        reached_interface |= (x >= hp.X_INTERFACE)
        reached_target |= (x >= hp.X_TARGET)
        exited = alive & (x >= hp.X_MAX)
        if np.any(exited):
            penetrated_count += int(exited.sum())
            t_first_exit[exited] = t
            reached_exit[exited] = True
            alive[exited] = False
            trapped[exited] = False
            trap_time_left[exited] = 0.0
            run_streak[exited] = 0.0
            x[exited] = hp.X_MAX
        # Interface crossing
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
        attempted_bc = alive & (x_prev < hp.REG_B_END) & (x >= hp.REG_B_END)
        gate_fail_bc = attempted_bc & (run_streak < hp.RUN_REQ_BC_S)
        if hp.USE_PROB_PASS:
            p_bc = m.pass_prob(gamma_eff, Ib_local, hp.PASS_P0_BC)
            rand_fail_bc = attempted_bc & (rng.random(hp.N) > p_bc)
        else:
            rand_fail_bc = np.zeros(hp.N, dtype=bool)
        blocked_bc = gate_fail_bc | rand_fail_bc
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
        y_top_violate = alive & (y > hp.Y_MAX)
        if np.any(y_top_violate):
            y[y_top_violate] = hp.Y_MAX
            y_unwrap[y_top_violate] = hp.Y_MAX
        y_bottom_violate = alive & (y < hp.Y_MIN)
        if np.any(y_bottom_violate):
            y[y_bottom_violate] = hp.Y_MIN
            y_unwrap[y_bottom_violate] = hp.Y_MIN
        # Aggregate stats
        mean_x[k] = float(np.mean(x[alive])) if np.any(alive) else float(np.mean(x))
        cross_flux_int[k] = float(((x_prev < hp.X_INTERFACE) & (x >= hp.X_INTERFACE)).sum()) / max(float(hp.N), 1e-12) / hp.DT_S
        cross_flux[k] = float(((x_prev < hp.X_TARGET) & (x >= hp.X_TARGET)).sum()) / max(float(hp.N), 1e-12) / hp.DT_S
        frac_interface[k] = float(reached_interface.mean())
        frac_target[k] = float(reached_target.mean())
        msd[k] = float(np.mean((x - x0) ** 2 + (y_unwrap - y0) ** 2))
        frac_in_interface[k] = float(np.mean((m.is_int_ab(x) | m.is_int_bc(x)) & alive))
        mask_int = (m.is_int_ab(x) | m.is_int_bc(x)) & alive
        gamma_int_mean[k] = float(np.mean(gamma_eff[mask_int])) if np.any(mask_int) else np.nan
        counts_xt, _ = np.histogram(x[alive], bins=x_edges_xt)
        xt_density[k] = counts_xt.astype(np.float32)
        if (k % max(1, hp.STATE_SAMPLE_STRIDE) == 0) and np.any(alive):
            counts_xy, _, _ = np.histogram2d(
                y[alive], x[alive], bins=[y_edges_xy, x_edges_xy]
            )
            occupancy_xy += counts_xy.astype(np.float32)
        traj_x[:, k + 1] = x[plot_idx]
        traj_y[:, k + 1] = y[plot_idx]
        traj_y_unwrap[:, k + 1] = y_unwrap[plot_idx]
        if use_frame_export and ((k + 1) % frame_steps == 0):
            progress.update(1)
        elif not use_frame_export:
            progress.update(1)
    progress.close()
    # Post processing summary
    dir_smooth = m.moving_average(dir_t, window_steps)
    mean_x_smooth = m.moving_average(mean_x, window_steps)
    frac_interface_smooth = m.moving_average(frac_interface, window_steps)
    cross_flux_int_smooth = m.moving_average(cross_flux_int, window_steps)
    frac_target_smooth = m.moving_average(frac_target, window_steps)
    cross_flux_smooth = m.moving_average(cross_flux, window_steps)
    msd_smooth = m.moving_average(msd, window_steps)
    time_axis = (np.arange(steps) + 1) * hp.DT_S
    # Summary dictionary (simplified)
    forward_disp = float((x - x0).mean())
    frac_target_end = float(reached_target.mean())
    penetration_rate = float(reached_exit.mean())
    summary = {
        "penetration_rate_xmax": penetration_rate,
        "frac_target_end": frac_target_end,
        "forward_displacement_um_mean": forward_disp,
        "mean_lambda_last": float(np.mean(mean_lambda_t[-window_steps:])),
        "mean_pcw_last": float(np.mean(mean_pcw_t[-window_steps:])),
    }
    # Create plots
    if make_plots:
        run_dir.mkdir(parents=True, exist_ok=True) if run_dir is not None else None
        fig_tag = f"mode={hp.LIGHT_MODE}, mapping={hp.LIGHT_MAPPING}"
        # Figure 12: visualise CheY cascade on separate subplots for clarity
        fig12, axs12 = plt.subplots(5, 1, sharex=True, figsize=(10.0, 8.0))
        # First subplot: light input u(t)
        axs12[0].plot(time_axis, u_t, color="#006d2c")
        axs12[0].set_ylabel("u(t)")
        axs12[0].set_title("CheY cascade dynamics")
        # Second: mean CheY-P concentration
        axs12[1].plot(time_axis, mean_chey_p_t, color="#238b45")
        axs12[1].set_ylabel("[CheY–P]")
        # Third: mean P_CW
        axs12[2].plot(time_axis, mean_pcw_t, color="#41ab5d")
        axs12[2].set_ylabel("P_CW")
        # Fourth: mean λ
        axs12[3].plot(time_axis, mean_lambda_t, color="#74c476")
        axs12[3].set_ylabel("λ")
        # Fifth: fraction of tumbling cells
        axs12[4].plot(time_axis, frac_tumble, color="#a1d99b")
        axs12[4].set_ylabel("tumble frac")
        axs12[4].set_xlabel("time (s)")
        for ax_sub in axs12:
            ax_sub.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
        fig12.tight_layout(h_pad=0.3)
        if save_outputs and run_dir is not None:
            fig12.savefig(run_dir / "12_chey_cascade_timeseries.png", dpi=300)
        plt.close(fig12)
        # Figure 13: pulse response
        t_resp = np.linspace(0.0, 15.0, 300)
        chey_resp = np.zeros_like(t_resp)
        pcw_resp = np.zeros_like(t_resp)
        chey_c = hp.CHEY_P_BASE
        dstep = t_resp[1] - t_resp[0]
        resp_delay_steps = max(1, int(round(hp.CHEY_DELAY_S / dstep)))
        resp_buf = np.zeros((resp_delay_steps,), dtype=float)
        for i, tt in enumerate(t_resp):
            I_eff_cell = 1.0 if (0.0 <= tt < 5.0) else 0.0
            chey_c = opto_chey_update(np.array([chey_c]), np.array([I_eff_cell]), dstep, hp)[0]
            resp_buf[i % resp_delay_steps] = chey_c
            chey_delayed = resp_buf[(i + 1) % resp_delay_steps]
            chey_resp[i] = chey_delayed
            pcw_resp[i] = chey_p_to_pcw(np.array([chey_delayed]), hp)[0]
        fig13, ax13 = plt.subplots(figsize=(10.0, 5.0))
        pulse = (t_resp < 5.0).astype(float)
        ax13.plot(t_resp, pulse, label="input pulse")
        ax13.plot(t_resp, chey_resp / hp.CHEY_P_MAX, label="[CheY–P] (norm)")
        ax13.plot(t_resp, pcw_resp, label="P_CW")
        ax13.set_title("Optogenetic pulse response (normalized)")
        ax13.set_xlabel("t (s)")
        ax13.set_ylabel("normalized units")
        ax13.legend(loc="upper right")
        ax13.grid(True, linestyle=":", linewidth=0.7, alpha=0.3)
        fig13.tight_layout()
        if save_outputs and run_dir is not None:
            fig13.savefig(run_dir / "13_pulse_response_delay.png", dpi=300)
        plt.close(fig13)
        # Figure 14: steady state curves
        I_vals = np.linspace(0.0, 1.0, 200)
        chey_ss = (hp.K_ACT / (hp.K_ACT + hp.K_DEACT)) * hp.CHEY_P_MAX * I_vals
        pcw_ss = chey_p_to_pcw(chey_ss, hp)
        lam_ss = pcw_to_lambda(pcw_ss, hp)
        fig14, ax14 = plt.subplots(figsize=(10.0, 5.0))
        ax14.plot(I_vals, pcw_ss, label="P_CW steady")
        ax14.plot(I_vals, (lam_ss - hp.LAMBDA_MIN) / (hp.LAMBDA_MAX - hp.LAMBDA_MIN), label="λ (norm)")
        ax14.set_title("Steady‑state P_CW and λ vs light intensity")
        ax14.set_xlabel("light intensity")
        ax14.set_ylabel("normalized units")
        ax14.legend(loc="upper left")
        ax14.grid(True, linestyle=":", linewidth=0.7, alpha=0.3)
        fig14.tight_layout()
        if save_outputs and run_dir is not None:
            fig14.savefig(run_dir / "14_pcw_light_response_curve.png", dpi=300)
        plt.close(fig14)
    return summary


if __name__ == "__main__":
    hp = m.HyperParams()
    # Increase number of agents and simulation time to make the CheY–P dynamics more apparent
    hp.N = 300
    hp.T_S = 100.0  # seconds
    # Use pulsed blue light with higher frequency and lower duty cycle to create multiple activation cycles
    hp.CHEY_PULSE_FREQ_HZ = 0.5
    hp.CHEY_PULSE_DUTY = 0.2
    hp.LIGHT_MAPPING = "chey"
    hp.LIGHT_MODE = "blue"
    # Disable animation and saving frames for faster execution
    hp.SAVE_FRAMES = False
    hp.MAKE_VIDEO = False
    hp.SHOW_PLOTS = False
    hp.OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs" / "stage1")
    hp.USE_OPTO_CHEY_DYNAMICS = True
    # Update global HP used by m.chey_u_of_t to match our local hyperparameters
    # This ensures that the pulse frequency and duty cycle are reflected when computing u(t)
    m.HP.CHEY_PULSE_FREQ_HZ = hp.CHEY_PULSE_FREQ_HZ
    m.HP.CHEY_PULSE_DUTY = hp.CHEY_PULSE_DUTY
    m.HP.CHEY_PULSE_PHASE_S = 0.0
    m.HP.CHEY_MODE = "pulse"
    simulate_stage1(hp, make_plots=True, save_outputs=True)