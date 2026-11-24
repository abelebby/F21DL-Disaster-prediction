# Maheen's ablations and error analysis

Run from the repository root:

```bash
python -m pip install -r requirements-baselines.txt -r requirements-ablation.txt
python src/run_baselines.py
python src/ablation_error_analysis.py
```

Existing checked-in baseline outputs can also be used without retraining. The default ablation command uses `data/frozen/`, `results/baselines/` and `results/ablation/`. Explicit equivalents are:

```bash
python src/ablation_error_analysis.py --frozen-dir data/frozen --baseline-dir results/baselines --results-dir results/ablation
```

The analysis preserves Maheen's modelling procedure: all three regression families, drop-group and single-group ablations, fold-specific high-flow thresholds, and errors by station and month. It reads Abel's selected settings and saved out-of-fold predictions. Default evaluation uses 2024/2025 validation folds, not the 2026 test set. `--final-test` explicitly evaluates the reserved test period; do not use it to select settings.

Outputs include `ablation_results.csv`, `baselines.csv`, `error_summary.csv`, station/month error tables, worst-prediction tables and five PNG figures. Runs overwrite outputs in the selected results directory; use `--results-dir /tmp/f21dl-ablation-check` for an isolated verification run.

MAE and RMSE in these ablation outputs are on the log-discharge scale. They must not be confused with original-scale MAE/RMSE in Abel's baseline table. The 90th-percentile high-flow labels are statistical proxies, not confirmed flood records.
