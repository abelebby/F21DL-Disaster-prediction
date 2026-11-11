# Abel's baseline models and clustering

Run from the repository root:

```bash
python src/run_baselines.py
```

Requires Python with numpy, pandas, scikit-learn, matplotlib and joblib. On Abel's Mac the existing `/opt/anaconda3/bin/python` has these packages.

The script reads the supplied train.csv, feature_config.json and cv_folds.py in data/frozen/. It checks the recorded checksums without changing Hann's files.

## What it does

- Fits linear regression, decision tree regression and random forest regression on both frozen feature sets.
- Uses Hann's two expanding-year validation folds. Scaling is fitted separately inside each fold for linear regression.
- Compares tree depths 5, 10 and unlimited, and forest minimum leaf sizes 1 and 5 (100 trees). Settings are chosen by mean fold RMSE on log1p discharge. These validation scores are development estimates used for selection, not an unbiased final evaluation.
- Fits the chosen settings to all training rows and saves models.
- Includes persistence and station/month climatology as reference predictions.
- Clips predicted log discharge below zero to zero consistently. Reports log RMSE and original-unit MAE/RMSE and R-squared. Back-transforming log predictions does not guarantee unbiased estimates of mean discharge.
- Runs exploratory K-means on the nine weather/rain-history columns from the training data, excluding station indicators, seasonal indicators, discharge history and the target. Standardises the inputs, tries k=2 through 6 and selects by silhouette on a reproducible 2,000-row sample. These are exploratory weather groups, not verified flood categories. PCA is only a two-dimensional display; clustering uses all nine scaled inputs.

## Files to share

Outputs are in results/baselines:

- validation_summary.csv and validation_scores.csv: model comparison and individual fold scores.
- validation_predictions.csv: dated, station-linked predictions for Maheen. Contains actual/predicted values in both log and original units.
- parameter_search.csv and selected_settings.json: the settings tried and chosen.
- *.joblib: models refitted on all training data (use validation predictions, not these final models, to assess historical validation rows).
- cluster_profiles.csv: average weather conditions and size of each group.
- cluster_assignments.csv and cluster_selection.csv: group membership and selection evidence.
- validation_comparison.png and clusters.png: figures.
- run_info.json: seed, software version and input checksums.

The rain_only set includes other weather, season and station features; its name means no discharge history.

## Final evaluation

The default command does not score 2026. When the group has fixed its modelling decisions, run:

```bash
python src/run_baselines.py --final-test
```

This also exports test_scores.csv and test_predictions.csv. Do not tune settings based on these test scores. Rerunning uses the same deterministic development procedure; generated output files are overwritten.

## Interpretation limits

The supplied inputs include day-t observed weather and predict day-t discharge. Describe this as same-day estimation, not operational next-day forecasting. Historical discharge in validation/test features assumes that actual previous-day observations become available each day; this is not an entire year forecast made in advance.

Only ten known locations and a few years are covered. Overall scores can hide poor results at individual locations or during peaks. Dataset measurement provenance remains unverified. The prepared features were chosen using 2023–2025, including the validation years, so validation is not a fully independent assessment of feature selection. The reserved 2026 test remains the final check.
