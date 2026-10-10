from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

import stage3_dual_light_strategy as s3


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "light_vs_dark_three_group"
SEEDS = [7, 44, 81]
BASE_FORWARD_KAPPA = 7.5
ORIG_BASE_HP = s3.base_hp
LIGHT_EFFECT_SCALE = "conservative"

STRATEGIES: Dict[str, Dict[str, float | str]] = {
    "dark_no_light": {
        "source_strategy": "off_control",
        "label": "No light",
    },
    "blue_baseline": {
        "source_strategy": "baseline_blue_0p5_0p2",
        "label": "Blue baseline",
        "freq": 1.00,
        "duty": 1.00,
    },
    "optimized_primed_push": {
        "source_strategy": "recommended_primed_push",
        "label": "Optimized light",
        "scan_s": 0.2,
        "scan_freq": 0.70,
        "scan_duty": 0.06,
        "push_freq": 0.25,
        "push_duty": 0.25,
    },
}


def base_hp(seed: int) -> s3.m.HyperParams:
    hp = ORIG_BASE_HP(seed)
    hp.BASE_FORWARD_KAPPA = BASE_FORWARD_KAPPA
    hp.USE_PULSE_DARK_PENALTY = True
    hp.PHOTO_BIAS_KAPPA_GAIN = 0.0
    hp.PASS_BETA_BLUE = 0.0
    hp.V_BLUE_GAIN = 0.0
    hp.TRAP_BLUE_REDUCE = 0.0
    return hp


def write_rows(path: Path, rows: Iterable[Dict[str, float | str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_timeseries(path: Path, strategy: str, timeseries: Dict[str, np.ndarray]) -> None:
    keys = list(timeseries.keys())
    rows: List[Dict[str, float | str]] = []
    for i in range(len(timeseries[keys[0]])):
        row: Dict[str, float | str] = {"strategy": strategy}
        for key in keys:
            row[key] = float(timeseries[key][i])
        rows.append(row)
    write_rows(path, rows)


def aggregate_rows(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    metrics = [
        "penetration_fraction",
        "frac_target_end",
        "forward_displacement_um_mean",
        "total_light_dose_s",
        "avg_gamma_abm",
        "swiping_fraction",
        "nosing_fraction",
        "first_passage_target_mean_s",
        "first_passage_exit_mean_s",
    ]
    out: List[Dict[str, float | str]] = []
    for name, strategy in STRATEGIES.items():
        subset = [r for r in rows if r["strategy"] == name]
        row: Dict[str, float | str] = {
            "strategy": name,
            "label": str(strategy["label"]),
            "n_runs": float(len(subset)),
        }
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in subset], dtype=float)
            finite = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_sd"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
        out.append(row)
    return out


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top")


def save_panel_jpegs(fig: plt.Figure, panel_axes: Dict[str, List[plt.Axes]], out_dir: Path) -> None:
    panel_dir = out_dir / "panel_jpg_600ppi"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for stem, axes in panel_axes.items():
        bboxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
        bbox_inches = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        fig.savefig(panel_dir / f"{stem}.jpg", dpi=600, bbox_inches=bbox_inches, facecolor="white")


def plot_three_group(summary_rows: List[Dict[str, float | str]], timeseries_by_strategy: Dict[str, Dict[str, np.ndarray]]) -> None:
    apply_style()
    colors = {
        "dark_no_light": "#BAD2E1",
        "blue_baseline": "#367DB0",
        "optimized_primed_push": "#3D9F3C",
    }
    labels = ["no light", "blue", "optimized\nlight"]
    order = list(STRATEGIES)
    row_by_strategy = {r["strategy"]: r for r in summary_rows}
    x = np.arange(len(order))

    fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    pen_mean = [float(row_by_strategy[k]["penetration_fraction_mean"]) for k in order]
    pen_sd = [float(row_by_strategy[k]["penetration_fraction_sd"]) for k in order]
    ax_a.bar(x, pen_mean, yerr=pen_sd, color=[colors[k] for k in order], edgecolor="#2F2F2F", linewidth=0.35, capsize=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("exit penetration fraction")
    ax_a.set_title("penetration fraction")
    ax_a.set_ylim(0, max(pen_mean) * 1.15)
    panel_label(ax_a, "a")

    exit_mean = [float(row_by_strategy[k]["first_passage_exit_mean_s_mean"]) for k in order]
    exit_sd = [float(row_by_strategy[k]["first_passage_exit_mean_s_sd"]) for k in order]
    ax_b.bar(x, exit_mean, yerr=exit_sd, color=[colors[k] for k in order], edgecolor="#2F2F2F", linewidth=0.35, capsize=2)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("mean exit time (s)")
    ax_b.set_title("penetration time")
    ax_b.set_ylim(min(exit_mean) - 12, max(exit_mean) + 8)
    panel_label(ax_b, "b")

    for name in order:
        ts = timeseries_by_strategy[name]
        ax_c.plot(ts["time_s"], ts["frac_exit_t"], color=colors[name], lw=1.4, label=str(STRATEGIES[name]["label"]))
    ax_c.set_xlabel("time (s)")
    ax_c.set_ylabel("cumulative exit fraction")
    ax_c.set_title("exit kinetics, seed 7")
    ax_c.set_xlim(0, max(float(ts["time_s"][-1]) for ts in timeseries_by_strategy.values()))
    ax_c.set_ylim(0, 1.03)
    ax_c.legend(loc="upper left")
    panel_label(ax_c, "c")

    stem = OUTPUT_DIR / "light_vs_dark_three_group"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".jpg"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    save_panel_jpegs(
        fig,
        {
            "panel_a_penetration_fraction": [ax_a],
            "panel_b_penetration_time": [ax_b],
            "panel_c_exit_kinetics": [ax_c],
        },
        OUTPUT_DIR,
    )
    plt.close(fig)


def write_summary(summary_rows: List[Dict[str, float | str]]) -> None:
    row = {r["strategy"]: r for r in summary_rows}
    dark = row["dark_no_light"]
    blue = row["blue_baseline"]
    opt = row["optimized_primed_push"]
    text = f"""Light-vs-dark three-group rerun
=================================

Purpose:
Only compare three groups: no light, blue baseline light, and optimized light.

Shared calibration:
BASE_FORWARD_KAPPA = {BASE_FORWARD_KAPPA}
This gives a no-light penetration fraction close to 20%, as requested.

Light-effect calibration:
USE_PULSE_DARK_PENALTY = True
PHOTO_BIAS_KAPPA_GAIN = 0.0
PASS_BETA_BLUE = 0.0
V_BLUE_GAIN = 0.0
TRAP_BLUE_REDUCE = 0.0
This conservative setting prevents pulsed light from producing unrealistically near-complete penetration.

Groups:
1. No light: LIGHT_MODE='off'.
2. Blue baseline: continuous blue light, implemented as 1.00 Hz, duty 1.00.
3. Optimized light: 0.2 s high-frequency priming (0.70 Hz, duty 0.06), then low-frequency push (0.25 Hz, duty 0.25).

Formal results, N=900, seeds={SEEDS}:
- No light penetration fraction = {float(dark['penetration_fraction_mean']):.4f} +/- {float(dark['penetration_fraction_sd']):.4f}
- Blue baseline penetration fraction = {float(blue['penetration_fraction_mean']):.4f} +/- {float(blue['penetration_fraction_sd']):.4f}
- Optimized light penetration fraction = {float(opt['penetration_fraction_mean']):.4f} +/- {float(opt['penetration_fraction_sd']):.4f}

- No light mean exit time = {float(dark['first_passage_exit_mean_s_mean']):.2f} +/- {float(dark['first_passage_exit_mean_s_sd']):.2f} s
- Blue baseline mean exit time = {float(blue['first_passage_exit_mean_s_mean']):.2f} +/- {float(blue['first_passage_exit_mean_s_sd']):.2f} s
- Optimized light mean exit time = {float(opt['first_passage_exit_mean_s_mean']):.2f} +/- {float(opt['first_passage_exit_mean_s_sd']):.2f} s

Main conclusion:
Compared with no light, optimized light increases penetration by {(float(opt['penetration_fraction_mean']) - float(dark['penetration_fraction_mean'])) * 100:.1f} percentage points and shortens mean exit time by {float(dark['first_passage_exit_mean_s_mean']) - float(opt['first_passage_exit_mean_s_mean']):.2f} s.
"""
    (OUTPUT_DIR / "LIGHT_VS_DARK_SUMMARY.txt").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s3.base_hp = base_hp
    all_rows: List[Dict[str, float | str]] = []
    timeseries_by_strategy: Dict[str, Dict[str, np.ndarray]] = {}

    for name, strategy in STRATEGIES.items():
        source_name = str(strategy["source_strategy"])
        source_strategy = dict(s3.STRATEGIES[source_name])
        for key, value in strategy.items():
            if key not in {"source_strategy", "label"}:
                source_strategy[key] = value
        for seed in SEEDS:
            capture = seed == SEEDS[0]
            summary, ts = s3.simulate_strategy(name, source_strategy, seed=seed, capture_timeseries=capture)
            summary["strategy"] = name
            summary["label"] = str(strategy["label"])
            all_rows.append(summary)
            if capture:
                timeseries_by_strategy[name] = ts
                save_timeseries(OUTPUT_DIR / f"timeseries_{name}_seed{seed}.csv", name, ts)
            print(
                f"{name:24s} seed={seed:3d} "
                f"pen={float(summary['penetration_fraction']):.4f} "
                f"exit_t={float(summary['first_passage_exit_mean_s']):.2f}"
            )

    summary_rows = aggregate_rows(all_rows)
    write_rows(OUTPUT_DIR / "all_runs.csv", all_rows)
    write_rows(OUTPUT_DIR / "strategy_summary.csv", summary_rows)
    (OUTPUT_DIR / "strategy_definitions.json").write_text(json.dumps(STRATEGIES, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "shared_hyperparams.json").write_text(
        json.dumps(
            {
                "BASE_FORWARD_KAPPA": BASE_FORWARD_KAPPA,
                "USE_PULSE_DARK_PENALTY": True,
                "PHOTO_BIAS_KAPPA_GAIN": 0.0,
                "PASS_BETA_BLUE": 0.0,
                "V_BLUE_GAIN": 0.0,
                "TRAP_BLUE_REDUCE": 0.0,
                "LIGHT_EFFECT_SCALE": LIGHT_EFFECT_SCALE,
                "seeds": SEEDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_three_group(summary_rows, timeseries_by_strategy)
    write_summary(summary_rows)


if __name__ == "__main__":
    main()
