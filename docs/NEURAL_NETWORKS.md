# Louis's MLP and 2D-CNN: next-day high-flow classification

The canonical scripts are `MLP code.py` and `2D-CNN code.py`. The old `MLP.py`, `CNN.py` and image-based `CNN 2D.py` were removed with permission. They remain available in Git history. The replacements use the existing cleaned weather CSV; no images are needed.

## Run

From the repository root, in a separate Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-neural.txt
python "MLP code.py"
python "2D-CNN code.py"
python src/run_high_flow_baselines.py
python src/summarize_high_flow.py
python -m unittest discover -s tests -v
```

The classifiers can also run using `requirements-baselines.txt` alone. Scripts resolve the data path relative to the repository. Neural scripts accept `--results-dir` and `--epochs` (default 100 with early stopping). A shortened epoch run is only a smoke test, not a report result.

## Shared setup

`src/high_flow_experiment.py` prepares station-specific daily features and tomorrow's discharge. Both networks and both sets of traditional classifiers use identical dated rows and labels. High flow is tomorrow's discharge exceeding the station's training-only 95th percentile. This is a statistical proxy, not evidence of actual inundation.

The original chronological boundaries are retained. A boundary day's rows are removed from training and validation so their target dates precede the next period's input dates:

| Split | Input dates | Rows | Latest target date |
| --- | --- | ---: | --- |
| Training | 7 January 2023–6 May 2025 | 8,510 | 7 May 2025 |
| Validation | 8 May–5 December 2025 | 2,120 | 6 December 2025 |
| Test | 7 December 2025–30 August 2026 | 2,670 | 31 August 2026 |

Thresholds and standardisation use training data only. All models select their classification cutoff by maximum validation F1. The networks retain Louis's architectures, Adam learning rate, early-stopping settings and validation weighting to training prevalence. Validation-weighted loss is not the natural validation-period average loss; the cutoff uses unweighted validation labels. Early stopping reduces overfitting risk but does not prove its absence. Seeds are fixed and deterministic TensorFlow operations requested, but software/hardware differences can still affect results.

The MLP uses 12 features. The CNN uses a 4-day × 4-variable grid. Logistic regression, a decision tree (depth 5, minimum leaf size 5), and random forest (100 trees, minimum leaf size 5) are fitted separately to each corresponding input representation. These classifier settings are fixed rather than tuned on test results. Persistence predicts high flow tomorrow when today's discharge already exceeds the station threshold. Its ranking score is the discharge/threshold ratio, not a probability.

## Outputs and comparison

- `results/high_flow/mlp/` and `results/high_flow/cnn/`: validation/test metrics and dated predictions, training history, loss figure, station thresholds and run metadata.
- Locally generated `model.keras` and `preprocessing.joblib`: model and fitted scaler; ignored by Git and recreated by running the scripts.
- `results/high_flow/classifiers/`: matched classifier and persistence results, with a combined `comparison.csv`.
- `results/high_flow/comparison.csv`: combined table including the networks, when generated after all runs.

Average precision (AP) is reported using `average_precision_score`; it is not trapezoidal PR-AUC. Compare AP, precision, recall, F1 and onset recall on the same test rows. There are few positive test cases, so small changes in predictions can materially change these metrics. Different network representations mean the MLP-versus-CNN comparison also changes input features, not architecture alone. Use each network's matching classifier feature set for the closest comparison.

These results must not be ranked against Abel's original same-day regression scores: the targets, evaluation periods and metrics differ. The original baseline, clustering and Maheen ablation files are untouched. Boundary correction, consistent date/station ordering and seed handling mean the revised networks must be rerun; do not reuse Louis's earlier numerical results as results of this revision.

GenAI assisted integration, shared preparation, exports, matched classifiers and validation checks. Louis's original model implementations are retained in the replacement scripts.

## Verified integration run — 25 November 2026

The MLP completed 41 epochs and the CNN 60 epochs using early stopping (maximum 100). Both saved models were reloaded and reproduced their exported test scores within numerical tolerance. Both used TensorFlow 2.18.1; the exact numerical-package versions are recorded in `results/high_flow/environment.json`. Four automated checks passed for split separation, next-day target alignment, training-only thresholds/scaling and cutoff selection. The comparison builder verified identical rows, labels, station thresholds and split metadata across all nine variants.

| Model / inputs | Average precision | Precision | Recall | F1 | Onsets caught / 29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mlp | 0.431 | 0.330 | 0.723 | 0.453 | 16 |
| cnn | 0.447 | 0.311 | 0.702 | 0.431 | 15 |
| mlp__logistic | 0.407 | 0.258 | 0.660 | 0.371 | 13 |
| mlp__tree | 0.335 | 0.419 | 0.553 | 0.477 | 8 |
| mlp__forest | 0.402 | 0.469 | 0.489 | 0.479 | 6 |
| cnn__logistic | 0.403 | 0.262 | 0.723 | 0.384 | 16 |
| cnn__tree | 0.224 | 0.360 | 0.574 | 0.443 | 11 |
| cnn__forest | 0.410 | 0.431 | 0.468 | 0.449 | 6 |
| persistence | 0.356 | 0.391 | 0.383 | 0.387 | 0 |

The CNN has the highest test average precision in this run; the random forest with MLP inputs has the highest F1. This is a descriptive comparison, not evidence of statistical significance or a reason to retune on the test set. There are only 47 positive test station-days and 29 onsets. These events are percentile-based high flow, not independently confirmed floods. The neural networks detect more onsets than either forest, with lower precision. These revised results supersede earlier network numbers for this corrected experiment; the original regression report numbers remain valid for their separate task.
