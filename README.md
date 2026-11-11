# Nepal river discharge modelling

Compare standard regression models and neural networks using prepared Nepal weather and river-flow data, and explore groups of similar weather conditions.

## Folder guide

| Location | Contents |
| --- | --- |
| `data/processed/` | Pad's cleaned CSV |
| `data/frozen/` | Hann's shared train/test CSVs, feature definitions and validation helper |
| `notebooks/01_split_features_freeze.ipynb` | Hann's data exploration, split and feature preparation notebook |
| `src/` | Downloading, cleaning, baseline training and clustering scripts |
| `results/baselines/` | Generated scores, predictions, plots and locally saved models |
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

## Shared prepared data

Use the existing files in `data/frozen/` for comparable experiments. The preparation notebook can regenerate them, but doing so overwrites the shared exports. Coordinate changes with the team. The notebook now reads the repository's cleaned CSV using its ISO date format and writes exports into `data/frozen/`.

The current target is log1p river discharge, not a confirmed flood label. Current features include same-day observed weather, so results describe same-day discharge estimation. Measurement provenance still needs verification.
