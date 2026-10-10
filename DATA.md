# Data and model weights

The current software does not include an ESM-2 model, model weights, or a web-service backend. None were found in the workspace inventory.

The ABM is parameterized in `src/merged_abm_full.py`; its default random seed and parameters are stored in the `HyperParams` class. The blue-light uniformity model generates its result files when run. Fish-region and Petri-dish simulations also generate output files when run.

Large data sets or model weights should be stored in a durable repository such as Zenodo and linked here with a version and DOI. No external data or weight link is currently available for this draft.

Existing historical result snapshots remain in the parent project folder. They are not copied here as benchmark evidence because their producing source revision has not been tied to this consolidated package.
