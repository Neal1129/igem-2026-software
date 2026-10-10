# ECUST-China iGEM 2026 Software

**Status: source-code organization draft.** This folder consolidates the software code currently available in the team workspace. It is not yet a release-ready submission: license selection, a compact reproducibility example, validation evidence, and the ESM-2 web source still need to be supplied.

## What the software contains

The primary application is **iGEM Motion Studio**, a Tkinter desktop interface for configuring and running an agent-based model (ABM) of bacterial motion through a mucus-like microfluidic barrier. The model includes light-response controls, mucus properties, flow and shear, trapping, interface crossing, and summary outputs.

Additional scripts cover blue-light uniformity, fish-region mucus transport, circular Petri-dish simulations, and exploratory model extensions. The exploratory scripts are retained separately from the primary application.

## Requirements

- Python 3.10 or later
- Tk/Tcl support in the Python installation to use the desktop interface
- Python packages listed in [`requirements.txt`](requirements.txt)

Install the Python dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Start the desktop application:

```bash
python src/merged_abm_desktop_app.py
```

Run the ABM directly:

```bash
python src/merged_abm_full.py
```

Run the independent simulation modules:

```bash
python src/blue_light_uniformity/model_sim.py
python src/models/circular_petri_dish_abm_48h.py
python src/models/fish_region_preview.py
```

The ABM and fish-region defaults write under `results/`. The blue-light model writes its generated data and figures beside its source file.

The historical stage experiments are run from the repository root. On macOS/Linux:

```bash
python research/experiments/stage1_script.py
python research/experiments/stage2_script.py
python research/experiments/stage3_script.py
python research/experiments/stage3_dual_light_strategy/stage3_dual_light_strategy.py
python research/experiments/stage3_dual_light_strategy/light_vs_dark_three_group.py
```

These stage scripts are research prototypes and have not been checked against the consolidated model after their imports were redirected.

## Repository map

| Path | Contents |
| --- | --- |
| `src/merged_abm_full.py` | Consolidated primary ABM model, selected from `merged-abm-full-V.py` |
| `src/merged_abm_desktop_app.py` | iGEM Motion Studio desktop application |
| `src/blue_light_uniformity/` | Blue-light field and uniformity simulation |
| `src/models/` | Fish-region preview and circular Petri-dish ABM |
| `research/experiments/` | Stage 1–3 exploratory scripts |
| `archive/legacy_sources/` | Earlier team-authored ABM and GUI versions, for provenance only |
| `scripts/` | Launch and desktop-build helpers |
| `docs/` | Architecture, inventory, data and validation notes |
| `examples/`, `results/`, `tests/` | Placeholders documenting reproducibility and validation evidence still to add |
| `web/` | Status note for the ESM-2 web application source, not found in the workspace |

## Current submission gaps

- No ESM-2 front-end or back-end source was present in the workspace.
- The team must choose an OSI-approved software license and add its full text as `LICENSE`. No license has been assigned in this draft.
- There is not yet a compact example input paired with an expected output and a recorded source version.
- Baseline result files exist elsewhere in the project, but their exact relationship to this consolidated source version has not been established. See [`docs/VALIDATION.md`](docs/VALIDATION.md).
- The desktop app and research scripts have not been run from this new folder.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Code inventory and provenance](docs/CODE_INVENTORY.md)
- [Data and model-weight policy](DATA.md)
- [Validation evidence and remaining work](docs/VALIDATION.md)
- [License decision](LICENSE_PENDING.md)

## Contributing

Keep the current application entry point and model module in `src/`. Put new reproducible research extensions in `research/experiments/`, and keep generated output out of source control. Before the official iGEM release, review changes on the `main` branch and record external or AI-assisted contributions in the project documentation.

## Authors

ECUST-China iGEM 2026 team. Individual code authors and third-party attributions should be completed by the team before release.

## License

Pending team selection. See [`LICENSE_PENDING.md`](LICENSE_PENDING.md); this draft does not grant reuse rights.

---

# ECUST-China iGEM 2026 软件代码

**状态：代码整理草稿。** 本目录汇集了当前团队工作区中能找到的软件源代码，但还不是可直接发布的最终提交：开源协议、最小可复现实例、验证证据和 ESM-2 网页源代码仍需补充。

核心软件 **iGEM Motion Studio** 使用 Tkinter 提供桌面界面，用于设置并运行细菌在仿生黏液屏障中的个体模型（ABM）。模型包含光响应、黏液性质、流场与剪切、困陷、界面穿越和结果汇总。蓝光均匀度、鱼体局部黏液穿透、培养皿模拟及阶段性扩展脚本也已归档。

安装、运行命令及目录说明见上文。`archive/legacy_sources/` 保存旧版本以供追溯；阶段实验属于探索性脚本，尚未在整理后的目录中运行核验。
