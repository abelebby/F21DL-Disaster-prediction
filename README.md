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
| `docs/ABLATION.md` | How to run Maheen's analysis using Abel's baseline outputs |
| `docs/ABEL_REPORT_SECTION.md` | Abel's report contribution |
| `MLP.py`, `CNN.py` | Louis's tabular neural-network experiments |
| `CNN 2D.py` | Separate image experiment requiring the external flood10k dataset |
| `docs/BASELINES.md` | Instructions and interpretation notes for Abel's work |
| `docs/legacy/` | Preserved original placeholder files |
| `requirements-baselines.txt` | Python dependencies for baselines and clustering |

## Run Abel's models

From this repository folder:

```bash
python -m pip install -r requirements-baselines.txt
python src/run_baselines.py
```

On my computer, `/opt/anaconda3/bin/python` has the required packages already installed.

This evaluates models on 2024/2025 validation folds and runs clustering on training data. It does not score the reserved 2026 test set by default. See [the baseline guide](docs/BASELINES.md) for the final-test command and output descriptions.

## Run Maheen's ablations and error analysis

From this repository folder, after running the baselines above:

```bash
python -m pip install -r requirements-ablation.txt
python src/ablation_error_analysis.py
```

The analysis reads `data/frozen/` and Abel's selected settings and validation predictions in `results/baselines/`. It writes to `results/ablation/`. Both contributors' checked-in results are retained. See [the ablation guide](docs/ABLATION.md).

## Other team contributions

Pad's cleaning script is `src/clean_data.py`; Hann's prepared data and notebook are listed above. Downloading raw data requires `kagglehub`; running the preparation notebook requires Jupyter. The existing cleaned and frozen data are sufficient for the baseline and ablation workflows; downloading raw data again is optional.

Louis's `MLP.py` and `CNN.py` read `data/processed/nepal_flood_clean.csv` and additionally require TensorFlow. They use their own feature construction, targets and splits, so their scores cannot be ranked directly against the frozen regression experiments. Despite its header, `CNN.py` uses Conv2D layers. `CNN 2D.py` is a separate image experiment: it requires an external `flood10k/` image directory that is not supplied in this repository and moves images into split folders when run.

The optional high-flow proxy work is preserved in `src/build_threshold_dataset.py` and `data/derived/thresholds/`. These labels are not confirmed flood observations and are separate from the regression report results.

The [repository integrity review](docs/REPOSITORY_INTEGRITY.md) records the history checks and integration repairs.

## Shared prepared data

Use the existing files in `data/frozen/` for comparable experiments. The preparation notebook can regenerate them, but doing so overwrites the shared exports. Coordinate changes with the team. The notebook now reads the repository's cleaned CSV using its ISO date format and writes exports into `data/frozen/`.

The current target is log1p river discharge, not a confirmed flood label. Current features include same-day observed weather, so results describe same-day discharge estimation. Measurement provenance still needs verification.

