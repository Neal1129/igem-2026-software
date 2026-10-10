# Software architecture

```mermaid
flowchart TD
    UI["Desktop UI<br/>Tkinter"] --> CORE["ABM core<br/>merged_abm_full.py"]
    CORE --> OUT[Run summaries\nplots and time series]
    BL[Blue-light uniformity model] --> BLO[Optical simulation results]
    FISH[Fish-region mucus model] --> FOUT[Spatial transport figures]
    PETRI[Circular Petri-dish ABM] --> POUT[48-hour comparison figure]
    EXP[Stage 1–3 research extensions] -. imports .-> CORE
    WEB[ESM-2 web component] -. source not present .-> CORE
```

The desktop application is the primary user-facing entry point. It imports the ABM module directly. The other simulation files are independent programs. Stage scripts extend or wrap the ABM for research experiments and are kept outside the primary application path.

This diagram describes the organized source tree. It does not claim that the stage experiments or the ESM-2 web component have been integrated or validated.
