from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("DISABLE_CJ_INSTRUMENTATION", "1")
os.environ.setdefault("CAAS_JUPYTER_SERVER_PORT", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MPL_CACHE = OUTPUT_DIR / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


def load_abm_module():
    """Load the root ABM file without requiring a copied merged_abm.py."""
    model_path = PROJECT_ROOT / "src" / "merged_abm_full.py"
    spec = importlib.util.spec_from_file_location("merged_abm_stage3_base", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABM module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m = load_abm_module()


# Stage-3 additions. They are attached dynamically to keep compatibility with
# the original HyperParams dataclass used by the root model.
setattr(m.HyperParams, "USE_OPTO_CHEY_DYNAMICS", True)
setattr(m.HyperParams, "K_ACT", 1.5)
setattr(m.HyperParams, "K_DEACT", 1.0)
setattr(m.HyperParams, "CHEY_P_BASE", 0.05)
setattr(m.HyperParams, "CHEY_P_MAX", 1.0)
setattr(m.HyperParams, "CHEY_KD", 0.35)
setattr(m.HyperParams, "CHEY_HILL_N", 3.0)
setattr(m.HyperParams, "CHEY_DELAY_S", 0.2)

setattr(m.HyperParams, "USE_GAMMA_ABM", True)
setattr(m.HyperParams, "GAMMA_ABM_HIGH", 1.25)
setattr(m.HyperParams, "GAMMA_ABM_LOW", 0.35)
setattr(m.HyperParams, "SWIPING_TUMBLE_SCALE", 1.25)
setattr(m.HyperParams, "NOSING_TUMBLE_SCALE", 0.55)

setattr(m.HyperParams, "USE_RHEOTAXIS", True)
setattr(m.HyperParams, "RHEO_GAIN", 0.65)
setattr(m.HyperParams, "RHEO_GAMMA_HALF", 1.0)
setattr(m.HyperParams, "RHEO_ST_GAIN", 1.5)
setattr(m.HyperParams, "THETA_FLOW", -np.pi / 2.0)
setattr(m.HyperParams, "BASE_FORWARD_KAPPA", 0.0)
setattr(m.HyperParams, "USE_PULSE_DARK_PENALTY", False)


STRATEGIES: Dict[str, Dict[str, float | str]] = {
    "off_control": {
        "label": "Off control",
        "kind": "off",
        "freq": 0.0,
        "duty": 0.0,
    },
    "baseline_blue_0p5_0p2": {
        "label": "Blue baseline",
        "kind": "fixed",
        "freq": 0.50,
        "duty": 0.20,
    },
    "push_lowfreq_highduty": {
        "label": "Low-freq push",
        "kind": "fixed",
        "freq": 0.25,
        "duty": 0.45,
    },
    "scan_highfreq_lowduty": {
        "label": "High-freq scan",
        "kind": "fixed",
        "freq": 0.55,
        "duty": 0.12,
    },
    "dual_phase_scan_push": {
        "label": "Dual periodic",
        "kind": "dual_periodic",
        "scan_freq": 0.55,
        "scan_duty": 0.12,
        "push_freq": 0.25,
        "push_duty": 0.45,
        "scan_s": 5.0,
        "push_s": 15.0,
    },
    "dual_sparse_scan_push": {
        "label": "Dual sparse-scan push",
        "kind": "dual_periodic",
        "scan_freq": 0.55,
        "scan_duty": 0.10,
        "push_freq": 0.25,
        "push_duty": 0.50,
        "scan_s": 1.5,
        "push_s": 18.5,
    },
    "recommended_primed_push": {
        "label": "Optimized primed-push",
        "kind": "dual_periodic",
        "scan_freq": 0.70,
        "scan_duty": 0.08,
        "push_freq": 0.18,
        "push_duty": 0.55,
        "scan_s": 0.2,
        "push_s": 999.0,
    },
    "dual_interface_gated": {
        "label": "Dual interface-gated",
        "kind": "dual_interface",
        "scan_freq": 0.55,
        "scan_duty": 0.12,
        "push_freq": 0.25,
        "push_duty": 0.45,
        "scan_s": 4.0,
        "push_s": 16.0,
        "interface_band_um": 28.0,
    },
}


def opto_chey_update(chey_p: np.ndarray, i_eff: np.ndarray, dt: float, hp: m.HyperParams) -> np.ndarray:
    d = hp.K_ACT * i_eff * (hp.CHEY_P_MAX - chey_p) - hp.K_DEACT * chey_p
    return np.clip(chey_p + d * dt, 0.0, hp.CHEY_P_MAX)


def chey_p_to_pcw(chey_p: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    return np.power(chey_p, hp.CHEY_HILL_N) / (
        np.power(hp.CHEY_KD, hp.CHEY_HILL_N) + np.power(chey_p, hp.CHEY_HILL_N)
    )


def pcw_to_lambda(pcw: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    return hp.LAMBDA_MIN + (hp.LAMBDA_MAX - hp.LAMBDA_MIN) * pcw


def rheotaxis_strength(gamma_eff: np.ndarray, mu_eff: np.ndarray, hp: m.HyperParams) -> np.ndarray:
    shear_part = gamma_eff / (hp.RHEO_GAMMA_HALF + gamma_eff + 1e-12)
    denom = max(hp.MU0_CY - hp.MU_INF, 1e-9)
    st_part = 1.0 + hp.RHEO_ST_GAIN * (hp.MU0_CY - mu_eff) / denom
    return hp.RHEO_GAIN * shear_part * st_part


def square_pulse(t: float, freq: float, duty: float) -> float:
    if freq <= 0.0 or duty <= 0.0:
        return 0.0
    phase = (t * freq) % 1.0
    return 1.0 if phase < duty else 0.0


def choose_light_phase(
    strategy: Dict[str, float | str],
    t: float,
    mean_x_live: float,
    hp: m.HyperParams,
) -> Tuple[float, float, float, str]:
    kind = str(strategy["kind"])
    if kind == "off":
        return 0.0, 0.0, 1.0, "off"
    if kind == "fixed":
        freq = float(strategy["freq"])
        duty = float(strategy["duty"])
        return square_pulse(t, freq, duty), freq, duty, "fixed"

    scan_freq = float(strategy["scan_freq"])
    scan_duty = float(strategy["scan_duty"])
    push_freq = float(strategy["push_freq"])
    push_duty = float(strategy["push_duty"])
    scan_s = float(strategy["scan_s"])
    push_s = float(strategy["push_s"])
    cycle = scan_s + push_s

    phase = t % cycle
    in_scan_window = phase < scan_s
    if kind == "dual_interface":
        band = float(strategy["interface_band_um"])
        near_ab = abs(mean_x_live - hp.REG_A_END) <= band
        near_bc = abs(mean_x_live - hp.REG_B_END) <= band
        in_scan_window = in_scan_window or near_ab or near_bc

    if in_scan_window:
        return square_pulse(t, scan_freq, scan_duty), scan_freq, scan_duty, "scan"
    return square_pulse(t, push_freq, push_duty), push_freq, push_duty, "push"


def base_hp(seed: int) -> m.HyperParams:
    hp = m.HyperParams()
    hp.SEED = seed
    hp.N = 900
    hp.T_S = 240.0
    hp.DT_S = 0.1
    hp.N_PLOT = 80
    hp.SAVE_FRAMES = False
    hp.MAKE_VIDEO = False
    hp.SHOW_PLOTS = False
    hp.LIGHT_MAPPING = "chey"
    hp.USE_CHEMOTAXIS = False
    hp.USE_MUCUS_PROFILE = False
    hp.USE_MUCUS_PHYS = True
    hp.USE_SHEAR_DRIFT = False
    hp.USE_RHEOTAXIS = False
    hp.USE_GAMMA_ABM = True
    hp.GAMMA_ABM_HIGH = 1.25
    hp.GAMMA_ABM_LOW = 0.35
    hp.SWIPING_TUMBLE_SCALE = 1.25
    hp.NOSING_TUMBLE_SCALE = 0.55
    hp.PHOTO_BIAS_KAPPA_GAIN = 10.0
    hp.V_BLUE_GAIN = 0.09
    hp.TRAP_BLUE_REDUCE = 0.055
    hp.PASS_P0_AB = 0.55
    hp.PASS_P0_BC = 0.80
    hp.GAMMA_MULT = 1.0
    return hp


def simulate_strategy(
    strategy_name: str,
    strategy: Dict[str, float | str],
    *,
    seed: int,
    capture_timeseries: bool = False,
) -> Tuple[Dict[str, float | str], Dict[str, np.ndarray]]:
    hp = base_hp(seed)
    base_forward_kappa = float(getattr(hp, "BASE_FORWARD_KAPPA", 0.0))
    use_pulse_dark_penalty = bool(getattr(hp, "USE_PULSE_DARK_PENALTY", False))
    hp.LIGHT_MODE = "off" if strategy["kind"] == "off" else "blue"
    m.HP = m.HyperParams(**asdict(hp))
    hp = m.HP
    hp.BASE_FORWARD_KAPPA = base_forward_kappa
    hp.USE_PULSE_DARK_PENALTY = use_pulse_dark_penalty
    m.apply_mucus_layer_thickness(hp)

    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    x = rng.uniform(20.0, hp.REG_A_END - 20.0, size=hp.N)
    y = rng.uniform(hp.Y_MIN + 20.0, hp.Y_MAX - 20.0, size=hp.N)
    if hp.BASE_FORWARD_KAPPA > 0.0:
        theta = rng.vonmises(0.0, hp.BASE_FORWARD_KAPPA, size=hp.N)
    else:
        theta = rng.uniform(-np.pi, np.pi, size=hp.N)
    x0 = x.copy()
    y_unwrap = y.copy()
    run_streak = np.zeros(hp.N, dtype=float)
    trapped = np.zeros(hp.N, dtype=bool)
    trap_time_left = np.zeros(hp.N, dtype=float)
    alive = np.ones(hp.N, dtype=bool)
    reached_target = np.zeros(hp.N, dtype=bool)
    reached_exit = np.zeros(hp.N, dtype=bool)
    t_first_interface = np.full(hp.N, np.nan, dtype=float)
    t_first_target = np.full(hp.N, np.nan, dtype=float)
    t_first_exit = np.full(hp.N, np.nan, dtype=float)

    release_prob = 1.0 - np.exp(-hp.TRAP_RELEASE_RATE * hp.DT_S) if hp.USE_TRAP else 0.0
    chey_p = np.full(hp.N, hp.CHEY_P_BASE, dtype=float)
    delay_steps = max(1, int(round(hp.CHEY_DELAY_S / hp.DT_S)))
    chey_delay_buffer = np.zeros((delay_steps, hp.N), dtype=float)

    gamma_abm_t = np.zeros(steps, dtype=np.float32)
    mode_swipe_t = np.zeros(steps, dtype=np.float32)
    mode_nose_t = np.zeros(steps, dtype=np.float32)
    u_t = np.zeros(steps, dtype=np.float32)
    mean_x_t = np.zeros(steps, dtype=np.float32)
    frac_target_t = np.zeros(steps, dtype=np.float32)
    frac_exit_t = np.zeros(steps, dtype=np.float32)
    frac_trapped_t = np.zeros(steps, dtype=np.float32)
    mean_lambda_t = np.zeros(steps, dtype=np.float32)
    mean_pcw_t = np.zeros(steps, dtype=np.float32)
    phase_code_t = np.zeros(steps, dtype=np.float32)

    total_light = 0.0
    attempted_ab_count = 0
    passed_ab_count = 0
    attempted_bc_count = 0
    passed_bc_count = 0

    for k in range(steps):
        t_now = (k + 1) * hp.DT_S
        x_prev = x.copy()
        live_mask = alive
        mean_x_live = float(np.mean(x[live_mask])) if np.any(live_mask) else float(np.mean(x))
        u, phase_freq, phase_duty, phase_name = choose_light_phase(strategy, t_now, mean_x_live, hp)
        if hp.LIGHT_MODE == "off":
            u = 0.0
        u_t[k] = u
        phase_code_t[k] = {"off": 0.0, "fixed": 1.0, "push": 2.0, "scan": 3.0}[phase_name]

        ib_local = np.full(hp.N, u, dtype=float)
        ib_local[~alive] = 0.0
        chey_p[:] = opto_chey_update(chey_p, ib_local, hp.DT_S, hp)
        chey_delay_buffer[k % delay_steps, :] = chey_p
        chey_p_delayed = chey_delay_buffer[(k + 1) % delay_steps, :]
        pcw = chey_p_to_pcw(chey_p_delayed, hp)
        lam = pcw_to_lambda(pcw, hp)
        if hp.LIGHT_MODE == "off":
            lam = np.clip(lam + hp.LAMBDA_DARK_BOOST, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
        elif hp.USE_PULSE_DARK_PENALTY:
            dark_weight = 1.0 - m.hill01(np.clip(ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            lam = np.clip(lam + hp.LAMBDA_DARK_BOOST * dark_weight, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)

        mean_pcw = float(np.mean(pcw[alive])) if np.any(alive) else float(np.mean(pcw))
        mean_pcw_t[k] = mean_pcw
        if hp.USE_GAMMA_ABM and hp.LIGHT_MODE != "off":
            mean_pccw = max(1e-9, 1.0 - mean_pcw)
            duty_cycle = max(float(phase_duty), 1e-6)
            gamma_abm = (mean_pcw * float(phase_freq)) / (mean_pccw * duty_cycle)
            gamma_abm_t[k] = gamma_abm
            if gamma_abm > hp.GAMMA_ABM_HIGH:
                lam = lam * hp.SWIPING_TUMBLE_SCALE
                mode_swipe_t[k] = 1.0
            elif gamma_abm < hp.GAMMA_ABM_LOW:
                lam = lam * hp.NOSING_TUMBLE_SCALE
                mode_nose_t[k] = 1.0

        ux = m.ux_field(x, y, t_now)
        uy = m.uy_field(x, y, t_now)
        gamma_eff = m.shear_rate_field(x, y, t_now)
        mu_eff = m.mu_carreau_yasuda(gamma_eff)

        if hp.USE_TRAP:
            releasing = trapped & alive & (rng.random(hp.N) < release_prob)
            trapped[releasing] = False
            trap_time_left[releasing] = 0.0
            p_trap_base = m.trap_prob_base_by_region(x)
            if hp.LIGHT_MODE == "off":
                p_trap_base = p_trap_base * hp.TRAP_DARK_MULT
            elif hp.LIGHT_MODE in {"blue", "dual"}:
                phi_b = m.hill01(np.clip(ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                if hp.USE_PULSE_DARK_PENALTY:
                    p_trap_base = p_trap_base * (1.0 + (hp.TRAP_DARK_MULT - 1.0) * (1.0 - phi_b))
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
            trapped[entering] = True
            trap_time_left[entering] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering.sum())

        p_tumble = 1.0 - np.exp(-lam * hp.DT_S)
        tumble = alive & (~trapped) & (rng.random(hp.N) < p_tumble)
        run_mask = alive & (~trapped) & (~tumble)
        run_streak[trapped | tumble] = 0.0
        run_streak[run_mask] += hp.DT_S

        if hp.USE_RHEOTAXIS:
            g_rheo = rheotaxis_strength(gamma_eff, mu_eff, hp)
            theta[run_mask] += g_rheo[run_mask] * np.sin(hp.THETA_FLOW - theta[run_mask]) * hp.DT_S

        v0 = m.v0_by_region(x)
        dr0 = m.dr_by_region(x)
        dt0 = m.dt_by_region(x)
        if hp.USE_MUCUS_PROFILE:
            v0, dr0, dt0 = m.apply_mucus_scaling(v0, dr0, dt0, x)
        if hp.USE_MUCUS_PHYS:
            v0, dr0, _ = m.apply_mucus_phys(v0, dr0, np.ones_like(x), x, gamma_eff)
        if hp.LIGHT_MODE in {"blue", "dual"}:
            phi_b = m.hill01(np.clip(ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            if hp.USE_PULSE_DARK_PENALTY:
                v0 = v0 * (hp.V0_DARK_FACTOR + (1.0 - hp.V0_DARK_FACTOR) * phi_b)
            v0 = v0 * (1.0 + hp.V_BLUE_GAIN * phi_b)
        if hp.LIGHT_MODE == "off":
            v0 = v0 * hp.V0_DARK_FACTOR

        theta[run_mask] += np.sqrt(2.0 * dr0[run_mask] * hp.DT_S) * rng.standard_normal(run_mask.sum())
        idx_tumble = np.where(tumble)[0]
        if idx_tumble.size:
            if hp.LIGHT_MODE in {"blue", "dual"}:
                phi_light = m.hill01(np.clip(ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                kappa = hp.BASE_FORWARD_KAPPA + hp.PHOTO_BIAS_KAPPA_BASE + hp.PHOTO_BIAS_KAPPA_GAIN * phi_light
                kappa = np.clip(kappa, 0.0, hp.PHOTO_BIAS_KAPPA_MAX)
                theta[idx_tumble] = rng.vonmises(0.0, kappa[idx_tumble], size=idx_tumble.size)
            elif hp.BASE_FORWARD_KAPPA > 0.0:
                theta[idx_tumble] = rng.vonmises(0.0, hp.BASE_FORWARD_KAPPA, size=idx_tumble.size)
            else:
                theta[idx_tumble] = rng.uniform(-np.pi, np.pi, size=idx_tumble.size)

        speed = v0.copy()
        if hp.TUMBLE_NO_TRANSLATION:
            speed[tumble] = 0.0
        move_mask = alive & (~trapped)
        x[move_mask] += (ux[move_mask] + speed[move_mask] * np.cos(theta[move_mask])) * hp.DT_S
        y_unwrap[move_mask] += (uy[move_mask] + speed[move_mask] * np.sin(theta[move_mask])) * hp.DT_S
        y[move_mask] = ((y_unwrap[move_mask] - hp.Y_MIN) % (hp.Y_MAX - hp.Y_MIN)) + hp.Y_MIN

        newly_interface = alive & np.isnan(t_first_interface) & (x >= hp.X_INTERFACE)
        t_first_interface[newly_interface] = t_now
        newly_target = alive & np.isnan(t_first_target) & (x >= hp.X_TARGET)
        t_first_target[newly_target] = t_now
        reached_target |= (x >= hp.X_TARGET)

        exited = alive & (x >= hp.X_MAX)
        if np.any(exited):
            reached_exit[exited] = True
            t_first_exit[exited] = t_now
            alive[exited] = False
            trapped[exited] = False
            x[exited] = hp.X_MAX

        attempted_ab = alive & (x_prev < hp.REG_A_END) & (x >= hp.REG_A_END)
        attempted_ab_count += int(attempted_ab.sum())
        gate_fail_ab = attempted_ab & (run_streak < hp.RUN_REQ_AB_S)
        if hp.USE_PROB_PASS:
            p_ab = m.pass_prob(gamma_eff, ib_local, hp.PASS_P0_AB)
            rand_fail_ab = attempted_ab & (rng.random(hp.N) > p_ab)
        else:
            rand_fail_ab = np.zeros(hp.N, dtype=bool)
        blocked_ab = gate_fail_ab | rand_fail_ab
        passed_ab_count += int((attempted_ab & (~blocked_ab)).sum())
        if np.any(blocked_ab):
            x[blocked_ab] = hp.REG_A_END - hp.X_INT_EPS

        attempted_bc = alive & (x_prev < hp.REG_B_END) & (x >= hp.REG_B_END)
        attempted_bc_count += int(attempted_bc.sum())
        gate_fail_bc = attempted_bc & (run_streak < hp.RUN_REQ_BC_S)
        if hp.USE_PROB_PASS:
            p_bc = m.pass_prob(gamma_eff, ib_local, hp.PASS_P0_BC)
            rand_fail_bc = attempted_bc & (rng.random(hp.N) > p_bc)
        else:
            rand_fail_bc = np.zeros(hp.N, dtype=bool)
        blocked_bc = gate_fail_bc | rand_fail_bc
        passed_bc_count += int((attempted_bc & (~blocked_bc)).sum())
        if np.any(blocked_bc):
            idx_bc = np.where(blocked_bc)[0]
            backstep = rng.uniform(hp.BC_FAIL_BACKSTEP_MIN_UM, hp.BC_FAIL_BACKSTEP_MAX_UM, size=idx_bc.size)
            x_new = np.minimum(x[idx_bc] - backstep, hp.REG_B_END - hp.X_INT_EPS)
            x[idx_bc] = np.maximum(hp.REG_B_END - hp.DELTA_INT_BC, x_new)

        total_light += float(u) * hp.DT_S
        mean_x_t[k] = float(np.mean(x[alive])) if np.any(alive) else float(np.mean(x))
        frac_target_t[k] = float(reached_target.mean())
        frac_exit_t[k] = float(reached_exit.mean())
        frac_trapped_t[k] = float((alive & trapped).sum()) / max(int(alive.sum()), 1)
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))

    summary: Dict[str, float | str] = {
        "strategy": strategy_name,
        "label": str(strategy["label"]),
        "seed": float(seed),
        "N": float(hp.N),
        "T_s": float(hp.T_S),
        "penetration_fraction": float(reached_exit.mean()),
        "frac_target_end": float(reached_target.mean()),
        "forward_displacement_um_mean": float(np.mean(x - x0)),
        "total_light_dose_s": float(total_light),
        "avg_gamma_abm": float(np.nanmean(gamma_abm_t)),
        "swiping_fraction": float(np.mean(mode_swipe_t)),
        "nosing_fraction": float(np.mean(mode_nose_t)),
        "mean_lambda_last": float(np.mean(mean_lambda_t[-max(1, int(5.0 / hp.DT_S)) :])),
        "mean_trapped_fraction_last": float(np.mean(frac_trapped_t[-max(1, int(5.0 / hp.DT_S)) :])),
        "first_passage_interface_mean_s": float(np.nanmean(t_first_interface)) if np.any(np.isfinite(t_first_interface)) else float("nan"),
        "first_passage_target_mean_s": float(np.nanmean(t_first_target)) if np.any(np.isfinite(t_first_target)) else float("nan"),
        "first_passage_exit_mean_s": float(np.nanmean(t_first_exit)) if np.any(np.isfinite(t_first_exit)) else float("nan"),
        "attempted_ab": float(attempted_ab_count),
        "passed_ab": float(passed_ab_count),
        "attempted_bc": float(attempted_bc_count),
        "passed_bc": float(passed_bc_count),
    }
    timeseries = {
        "time_s": (np.arange(steps) + 1) * hp.DT_S,
        "u_t": u_t,
        "gamma_abm_t": gamma_abm_t,
        "mode_swipe_t": mode_swipe_t,
        "mode_nose_t": mode_nose_t,
        "mean_x_t": mean_x_t,
        "frac_target_t": frac_target_t,
        "frac_exit_t": frac_exit_t,
        "frac_trapped_t": frac_trapped_t,
        "mean_lambda_t": mean_lambda_t,
        "mean_pcw_t": mean_pcw_t,
        "phase_code_t": phase_code_t,
    }
    return summary, timeseries if capture_timeseries else {}


def write_rows(path: Path, rows: Iterable[Dict[str, float | str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    metrics = [
        "penetration_fraction",
        "frac_target_end",
        "forward_displacement_um_mean",
        "total_light_dose_s",
        "avg_gamma_abm",
        "swiping_fraction",
        "nosing_fraction",
        "mean_trapped_fraction_last",
        "first_passage_interface_mean_s",
        "first_passage_target_mean_s",
        "first_passage_exit_mean_s",
        "attempted_ab",
        "passed_ab",
        "attempted_bc",
        "passed_bc",
    ]
    out: List[Dict[str, float | str]] = []
    for name, strategy in STRATEGIES.items():
        subset = [r for r in rows if r["strategy"] == name]
        if not subset:
            continue
        agg: Dict[str, float | str] = {"strategy": name, "label": str(strategy["label"]), "n_runs": float(len(subset))}
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in subset], dtype=float)
            finite = vals[np.isfinite(vals)]
            agg[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            agg[f"{metric}_sd"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
        out.append(agg)
    return out


def save_timeseries(path: Path, strategy: str, timeseries: Dict[str, np.ndarray]) -> None:
    keys = list(timeseries.keys())
    rows = []
    n = len(timeseries[keys[0]])
    for i in range(n):
        row: Dict[str, float | str] = {"strategy": strategy}
        for key in keys:
            row[key] = float(timeseries[key][i])
        rows.append(row)
    write_rows(path, rows)


def apply_nature_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")


def save_panel_jpegs(fig: plt.Figure, panel_axes: Dict[str, List[plt.Axes]], out_dir: Path) -> None:
    panel_dir = out_dir / "panel_jpg_600ppi"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for stem, axes in panel_axes.items():
        bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
        bbox_inches = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        fig.savefig(
            panel_dir / f"{stem}.jpg",
            dpi=600,
            bbox_inches=bbox_inches,
            facecolor="white",
            edgecolor="none",
        )


def plot_nature_figure(summary_rows: List[Dict[str, float | str]], dual_ts: Dict[str, np.ndarray], out_dir: Path) -> None:
    apply_nature_style()
    # Palette adapted from the provided Nature-style green/blue reference.
    palette = {
        "green_dark": "#3D9F3C",
        "green_mid": "#9ED17B",
        "green_soft": "#AADCA9",
        "blue_dark": "#04579B",
        "blue_mid": "#367DB0",
        "blue_soft": "#9DC7DD",
        "cyan_mid": "#6CBAD8",
        "cyan_soft": "#DBF1FA",
        "teal_mid": "#6FC8CA",
        "neutral": "#BAD2E1",
        "neutral_dark": "#4D4D4D",
    }
    colors = {
        "off_control": palette["neutral"],
        "baseline_blue_0p5_0p2": palette["blue_mid"],
        "push_lowfreq_highduty": palette["green_mid"],
        "scan_highfreq_lowduty": palette["cyan_mid"],
        "dual_phase_scan_push": palette["blue_soft"],
        "dual_sparse_scan_push": palette["teal_mid"],
        "recommended_primed_push": palette["green_dark"],
        "dual_interface_gated": palette["neutral"],
    }
    order = [name for name in STRATEGIES if name != "dual_interface_gated"]
    row_by_strategy = {r["strategy"]: r for r in summary_rows}
    short_labels = {
        "off_control": "off",
        "baseline_blue_0p5_0p2": "blue\n0.5/0.2",
        "push_lowfreq_highduty": "push\n0.25/0.45",
        "scan_highfreq_lowduty": "scan\n0.55/0.12",
        "dual_phase_scan_push": "dual\nperiodic",
        "dual_sparse_scan_push": "dual\nsparse",
        "recommended_primed_push": "primed\n0.18/0.55",
    }
    labels = [short_labels[k] for k in order]
    x_pos = np.arange(len(order))

    fig = plt.figure(figsize=(7.35, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.02, 1.18, 1.18], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    # a: schedule schematic
    t = np.linspace(0, 40, 400)
    u_scan = np.array([square_pulse(tt, 0.55, 0.12) for tt in t])
    u_push = np.array([square_pulse(tt, 0.25, 0.45) for tt in t])
    u_dual = []
    dummy_hp = base_hp(7)
    for tt in t:
        u, _, _, phase = choose_light_phase(STRATEGIES["recommended_primed_push"], tt, 100.0, dummy_hp)
        u_dual.append(u + (0.06 if phase == "push" else 0.0))
    ax_a.plot(t, u_scan + 2.0, color=colors["scan_highfreq_lowduty"], lw=1.1)
    ax_a.plot(t, u_push + 1.0, color=colors["push_lowfreq_highduty"], lw=1.1)
    ax_a.plot(t, u_dual, color=colors["recommended_primed_push"], lw=1.35)
    ax_a.axvspan(0, 0.2, color=colors["recommended_primed_push"], alpha=0.10, linewidth=0)
    ax_a.annotate(
        "0.2 s priming",
        xy=(0.10, 3.08),
        xytext=(6.6, 3.08),
        arrowprops=dict(arrowstyle="-", color=colors["recommended_primed_push"], lw=0.55),
        fontsize=5.7,
        color=colors["recommended_primed_push"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.8),
        ha="left",
        va="center",
    )
    ax_a.set_yticks([2.5, 1.5, 0.5])
    ax_a.set_yticklabels(["scan", "push", "dual"])
    ax_a.set_xlabel("time (s)")
    ax_a.set_title("pulse programmes")
    ax_a.set_ylim(-0.15, 3.42)
    ax_a.set_xlim(0, 40)
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    panel_label(ax_a, "a")

    # b: penetration bars
    means = [float(row_by_strategy[k]["penetration_fraction_mean"]) for k in order]
    sds = [float(row_by_strategy[k]["penetration_fraction_sd"]) for k in order]
    ax_b.bar(x_pos, means, yerr=sds, color=[colors[k] for k in order], edgecolor="#2F2F2F", linewidth=0.35, capsize=2)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(labels, rotation=0, ha="center")
    ax_b.set_ylabel("exit penetration fraction")
    ax_b.set_title("primed-push reaches near-optimal penetration")
    ax_b.set_ylim(0, max(0.02, max(means) * 1.35))
    ax_b.margins(x=0.035)
    panel_label(ax_b, "b")

    # c: Pareto
    for k in order:
        row = row_by_strategy[k]
        ax_c.scatter(
            float(row["total_light_dose_s_mean"]),
            float(row["penetration_fraction_mean"]),
            s=42,
            color=colors[k],
            edgecolor="#2F2F2F",
            linewidth=0.35,
            zorder=3,
        )
    ax_c.set_xlabel("light dose (s)")
    ax_c.set_ylabel("exit penetration fraction")
    ax_c.set_title("penetration-dose trade-off")
    panel_label(ax_c, "c")

    # d: gamma and mode trace for proposed strategy
    t_dual = dual_ts["time_s"]
    ax_d.plot(t_dual, dual_ts["gamma_abm_t"], color=palette["blue_dark"], lw=0.9, label=r"$\gamma_{ABM}$")
    ax_d.axhline(base_hp(7).GAMMA_ABM_HIGH, color=palette["neutral_dark"], lw=0.6, ls="--")
    ax_d.axhline(base_hp(7).GAMMA_ABM_LOW, color=palette["neutral_dark"], lw=0.6, ls=":")
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("gamma_ABM")
    ax_d.set_title("duty-sensitive control state")
    ax_d.set_xlim(0, min(90, float(t_dual[-1])))
    ax_d2 = ax_d.twinx()
    ax_d2.fill_between(t_dual, 0, dual_ts["mode_swipe_t"], color=palette["cyan_mid"], alpha=0.32, linewidth=0)
    ax_d2.fill_between(t_dual, 0, -dual_ts["mode_nose_t"], color=palette["green_soft"], alpha=0.46, linewidth=0)
    ax_d2.set_ylim(-1.1, 1.1)
    ax_d2.set_yticks([-1, 0, 1])
    ax_d2.set_yticklabels(["nose", "", "swipe"])
    ax_d2.spines["top"].set_visible(False)
    panel_label(ax_d, "d")

    # e: target and exit accumulation trace
    ax_e.plot(t_dual, dual_ts["frac_target_t"], color=palette["blue_mid"], lw=1.2, label="target")
    ax_e.plot(t_dual, dual_ts["frac_exit_t"], color=colors["recommended_primed_push"], lw=1.2, label="exit")
    ax_e.set_xlabel("time (s)")
    ax_e.set_ylabel("cumulative fraction")
    ax_e.set_title("proposed strategy kinetics")
    ax_e.set_xlim(0, float(t_dual[-1]))
    ax_e.set_ylim(0, max(0.02, float(np.max(dual_ts["frac_target_t"])) * 1.2))
    ax_e.legend(loc="upper left")
    panel_label(ax_e, "e")

    stem = out_dir / "nature_dual_light_strategy"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    save_panel_jpegs(
        fig,
        {
            "panel_a_pulse_programmes": [ax_a],
            "panel_b_penetration_bars": [ax_b],
            "panel_c_penetration_dose": [ax_c],
            "panel_d_gamma_modes": [ax_d, ax_d2],
            "panel_e_strategy_kinetics": [ax_e],
        },
        out_dir,
    )
    plt.close(fig)


def write_contract(out_dir: Path) -> None:
    text = """Core conclusion:
Optimized primed-push dual-frequency pulsing uses a 0.2 s high-frequency priming pulse followed by long low-frequency propulsion (0.18 Hz, 0.55 duty), maximizing the combined high/low-frequency strategy while preserving the scan-to-push mechanism.

Figure archetype:
Quantitative grid with a compact schedule schematic and supporting model-output panels.

Target journal/output:
Nature-style working figure; SVG/PDF editable text plus high-resolution PNG/TIFF previews.

Backend:
Python/matplotlib only.

Final size:
Double-column style, 7.2 x 5.0 in.

Panel map:
a: pulse programme schematic for scan, push and dual schedules.
b: exit penetration fraction across strategies.
c: light dose versus penetration Pareto view.
d: gamma_ABM and swiping/nosing states for the proposed dual strategy.
e: target and exit accumulation kinetics for the proposed dual strategy.

Evidence hierarchy:
hero evidence: panel b penetration comparison.
validation evidence: panel c Pareto trade-off and panel e kinetics.
controls/robustness: off, fixed baseline, fixed push and fixed scan strategies.

Statistics:
Three random seeds per strategy; bars show mean and SD. This is a simulation comparison, not biological replication.

Source data:
all_runs.csv, strategy_summary.csv and representative_recommended_primed_push_timeseries.csv.

Reviewer risk:
The result is model-predicted and parameter-sensitive; next step should increase seeds and validate against the original blue/off baseline and microfluidic measurements.
"""
    (out_dir / "figure_contract.txt").write_text(text, encoding="utf-8")


def clean_outputs(out_dir: Path) -> None:
    for pattern in ("*.csv", "*.json", "*.txt", "*.svg", "*.pdf", "*.png", "*.tiff"):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)
    clean_outputs(OUTPUT_DIR)
    seeds = [7, 44, 81]
    all_rows: List[Dict[str, float | str]] = []
    representative_ts: Dict[str, np.ndarray] = {}
    for strategy_name, strategy in STRATEGIES.items():
        for seed in seeds:
            capture = strategy_name == "recommended_primed_push" and seed == seeds[0]
            summary, ts = simulate_strategy(strategy_name, strategy, seed=seed, capture_timeseries=capture)
            all_rows.append(summary)
            print(
                f"{strategy_name:24s} seed={seed:3d} "
                f"pen={float(summary['penetration_fraction']):.4f} "
                f"target={float(summary['frac_target_end']):.4f} "
                f"dose={float(summary['total_light_dose_s']):.1f} "
                f"swipe={float(summary['swiping_fraction']):.3f} "
                f"nose={float(summary['nosing_fraction']):.3f}"
            )
            if capture:
                representative_ts = ts

    summary_rows = aggregate_rows(all_rows)
    write_rows(OUTPUT_DIR / "all_runs.csv", all_rows)
    write_rows(OUTPUT_DIR / "strategy_summary.csv", summary_rows)
    if representative_ts:
        save_timeseries(OUTPUT_DIR / "representative_recommended_primed_push_timeseries.csv", "recommended_primed_push", representative_ts)
        plot_nature_figure(summary_rows, representative_ts, OUTPUT_DIR)
    write_contract(OUTPUT_DIR)
    (OUTPUT_DIR / "strategy_definitions.json").write_text(json.dumps(STRATEGIES, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
