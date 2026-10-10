# Code inventory and provenance

## Included project code

| Consolidated path | Original workspace path | Notes |
| --- | --- | --- |
| `src/merged_abm_full.py` | `merged-abm-full-V.py` | Chosen as the main ABM because the project notes identify it as the most complete model. Its default output directory now points inside this package. |
| `src/merged_abm_desktop_app.py` | `package_ready/merged_abm_desktop_app.py` | Packaged Tkinter GUI; imports `merged_abm_full`. |
| `src/blue_light_uniformity/model_sim.py` | `蓝光模型/model_sim.py` | The copy under `蓝光模型/blue_light_uniformity_model/` was byte-identical and was consolidated here once. |
| `src/models/fish_region_preview.py` | `0807/Fish1/preview.py` | Fish-region transport visualization; its default output path now points inside this package. |
| `src/models/circular_petri_dish_abm_48h.py` | `circular_petri_dish_48h/circular_petri_dish_abm_48h.py` | Independent 48-hour Petri-dish model. |
| `research/experiments/stage1_script.py` | `组会PPT/5.29组会/stage1_script.py` | Experimental CheY-P extension; import and output paths redirected. |
| `research/experiments/stage2_script.py` | `组会PPT/5.29组会/stage2_script.py` | Experimental rheotaxis/surface extension; import and output paths redirected. |
| `research/experiments/stage3_script.py` | `组会PPT/5.29组会/stage3_script.py` | Experimental strategy sweep; import and output paths redirected. |
| `research/experiments/stage3_dual_light_strategy/` | `组会PPT/5.29组会/stage3_dual_light_strategy/` | Stage 3 strategy scan and comparison script; its model path now points to `src/merged_abm_full.py`. |
| `archive/legacy_sources/` | Root, `package_ready/`, and `软著/软件代码/` snapshots | Earlier team ABM/GUI versions retained for provenance; not used by the current application entry point. |

The stage-script changes only update module names and output/model paths. Their scientific logic has not been revised or run from this package.

## Consolidated rather than copied

The root folder, `package_ready/`, and `软著/软件代码/` contained multiple ABM and GUI revisions. This package keeps one primary model and one desktop GUI in `src/`; earlier team versions are collected under `archive/legacy_sources/` and are not runtime dependencies. The two blue-light model copies were identical, so only one was included.

## Excluded items

- PyInstaller build directories, `.app`/`.exe` bundles, caches, and generated plots were not copied as source code.
- The third-party `bacterial-surface-adhesion-main` research checkout was not copied into this software package. Its source provenance/license needs separate review, and the directory contains a file named like a private SSH key.
- PDFs, papers, meeting documents, patent materials, and historical output collections remain in the parent project folder.
- No ESM-2 front-end/back-end files were present to include.
