# Nepal river discharge modelling

Compare standard regression models and neural networks using prepared Nepal weather and river-flow data, and explore groups of similar weather conditions.

## Folder guide

| Location | Contents |
| --- | --- |
| `data/processed/` | Pad's cleaned CSV |
| `data/frozen/` | Hann's shared train/test CSVs, feature definitions and validation helper |
| `notebooks/01_split_features_freeze.ipynb` | Hann's data exploration, split and feature preparation notebook |
| `src/` | Cleaning, baseline training, clustering, ablation and error-analysis scripts |
| `results/baselines/` | Generated scores, predictions, plots and locally saved models |
| `results/ablation/` | Maheen's ablation tables, error summaries and figures |
| `MLP code.py`, `2D-CNN code.py` | Louis's next-day high-flow neural networks |
| `src/run_high_flow_baselines.py` | Matched traditional classifiers for the neural comparison |
| `results/high_flow/` | Revised classification metrics, predictions and figures |
| `requirements-baselines.txt` | Python dependencies for baselines and clustering |

## Run Abel's models

From this repository folder:

```bash
python -m pip install -r requirements-baselines.txt
python src/run_baselines.py
```

On my computer, `/opt/anaconda3/bin/python` has the required packages already installed.

This evaluates models on 2024/2025 validation folds and runs clustering on training data. It does not score the reserved 2026 test set by default. Use `python src/run_baselines.py --final-test` to explicitly score the reserved test set. Outputs are saved in `results/baselines/`.

## Run Maheen's ablations and error analysis

From this repository folder, after running the baselines above:

```bash
python -m pip install -r requirements-ablation.txt
python src/ablation_error_analysis.py
```

The analysis reads `data/frozen/` and Abel's selected settings and validation predictions in `results/baselines/`. It writes to `results/ablation/`. Both contributors' checked-in results are retained.

## Run the networks and matching classifiers

```bash
python -m pip install -r requirements-neural.txt
python "MLP code.py"
python "2D-CNN code.py"
python src/run_high_flow_baselines.py
python src/summarize_high_flow.py
```

Outputs are saved in `results/high_flow/`. The combined table is `results/high_flow/comparison.csv`. These experiments predict next-day statistical high flow; their metrics cannot be ranked against the separate same-day regression scores.

## Other team contributions

Pad's cleaning script is `src/clean_data.py`; Hann's prepared data and notebook are listed above. Downloading raw data requires `kagglehub`; running the preparation notebook requires Jupyter. The existing cleaned and frozen data are sufficient for the baseline and ablation workflows; downloading raw data again is optional.

Louis's replacement scripts use the existing cleaned CSV and require no image dataset. Their MLP and 2D-CNN predict next-day statistical high flow; use the matched classifiers for direct comparison. This remains a separate experiment from the frozen same-day regression models. The superseded `MLP.py`, `CNN.py` and image-based `CNN 2D.py` have been removed with Louis's permission; previous versions remain in Git history.

The optional high-flow proxy work is preserved in `src/build_threshold_dataset.py` and `data/derived/thresholds/`. These labels are not confirmed flood observations and are separate from the regression report results.


## Shared prepared data

Use the existing files in `data/frozen/` for comparable experiments. The preparation notebook can regenerate them, but doing so overwrites the shared exports. Coordinate changes with the team. The notebook now reads the repository's cleaned CSV using its ISO date format and writes exports into `data/frozen/`.

The current target is log1p river discharge, not a confirmed flood label. Current features include same-day observed weather, so results describe same-day discharge estimation. Measurement provenance still needs verification.

