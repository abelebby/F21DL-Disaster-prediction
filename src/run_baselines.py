"""Baseline regression and exploratory clustering on Hann's frozen features.

AI assistance: implemented with ChatGPT at the user's request, 'Okay then lets
implement', following the baseline/clustering plan discussed with the user.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl_cache'))
FROZEN = ROOT / "data" / "frozen"
sys.path.insert(0, str(FROZEN))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from cv_folds import cv_splits

SEED = 42


def scores(y, pred):
    # Discharge cannot be negative; apply the same rule to every model.
    pred = np.maximum(np.asarray(pred), 0)
    raw, raw_pred = np.expm1(y), np.expm1(pred)
    return dict(rmse_log=float(np.sqrt(mean_squared_error(y, pred))),
                mae_m3s=float(mean_absolute_error(raw, raw_pred)),
                rmse_m3s=float(np.sqrt(mean_squared_error(raw, raw_pred))),
                r2_raw=float(r2_score(raw, raw_pred)))


def candidates():
    yield 'linear', {}, make_pipeline(StandardScaler(), LinearRegression())
    for depth in [5, 10, None]:
        yield 'tree', {'max_depth': depth, 'min_samples_leaf': 5}, DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=5, random_state=SEED)
    for leaf in [1, 5]:
        yield 'forest', {'n_estimators': 100, 'min_samples_leaf': leaf}, RandomForestRegressor(
            n_estimators=100, min_samples_leaf=leaf, random_state=SEED, n_jobs=1)


def export_predictions(frame, pred, model, feature_set, fold):
    out = frame[['date', 'location', 'river_discharge_m3s', 'y']].copy()
    out['prediction_log'] = np.maximum(pred, 0)
    out['prediction_m3s'] = np.expm1(out['prediction_log'])
    out['model'], out['feature_set'], out['fold'] = model, feature_set, fold
    return out


def reference(fit, hold, name):
    if name == 'persistence':
        return hold.log_q_lag1.to_numpy()
    means = fit.assign(month=fit.date.dt.month).groupby(['location', 'month']).y.mean()
    keys = pd.MultiIndex.from_arrays([hold.location, hold.date.dt.month])
    pred = means.reindex(keys).to_numpy()
    if not np.isfinite(pred).all():
        raise ValueError('Climatology has unseen station/month combinations')
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-test', action='store_true', help='Explicitly evaluate the reserved 2026 test set')
    args = parser.parse_args()
    out = ROOT / 'results' / 'baselines'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((FROZEN / 'feature_config.json').read_text())
    for filename, expected in config['file_sha256'].items():
        assert hashlib.sha256((FROZEN / filename).read_bytes()).hexdigest() == expected, filename + ' changed'
    train = pd.read_csv(FROZEN / 'train.csv', parse_dates=['date'])
    features = config['feature_sets']
    assert np.isfinite(train[features['with_discharge'] + ['y']]).all().all()
    assert not train.duplicated(['location', 'date']).any()
    folds = list(cv_splits(train))
    metrics, predictions, selected, fitted = [], [], {}, {}
    search_rows = []
    for feature_set, cols in features.items():
        print('Training:', feature_set, flush=True)
        trials = []
        for model_name, params, model in candidates():
            fold_metrics, fold_preds = [], []
            for fold, tr, va in folds:
                fit, hold = train.iloc[tr], train.iloc[va]
                model.fit(fit[cols], fit.y)
                pred = model.predict(hold[cols])
                fold_metrics.append(dict(model=model_name, feature_set=feature_set, fold=fold, **scores(hold.y, pred)))
                fold_preds.append(export_predictions(hold, pred, model_name, feature_set, fold))
            error = float(np.mean([m['rmse_log'] for m in fold_metrics]))
            trials.append((model_name, params, model, error, fold_metrics, fold_preds))
            search_rows.append(dict(model=model_name, feature_set=feature_set, params=json.dumps(params), mean_rmse_log=error))
        for name in ['linear', 'tree', 'forest']:
            winner = min((t for t in trials if t[0] == name), key=lambda t: t[3])
            _, params, model, error, ms, ps = winner
            metrics.extend(ms); predictions.extend(ps)
            key = feature_set + '__' + name
            selected[key] = dict(params=params, mean_validation_rmse_log=error)
            model.fit(train[cols], train.y)
            fitted[key] = (model, cols)
            joblib.dump(model, out / (key + '.joblib'))
    for name in ['persistence', 'climatology']:
        for fold, tr, va in folds:
            fit, hold = train.iloc[tr], train.iloc[va]
            pred = reference(fit, hold, name)
            metrics.append(dict(model=name, feature_set='reference', fold=fold, **scores(hold.y, pred)))
            predictions.append(export_predictions(hold, pred, name, 'reference', fold))
    pd.DataFrame(search_rows).to_csv(out / 'parameter_search.csv', index=False)
    pd.DataFrame(metrics).to_csv(out / 'validation_scores.csv', index=False)
    pd.concat(predictions).to_csv(out / 'validation_predictions.csv', index=False)
    (out / 'selected_settings.json').write_text(json.dumps(selected, indent=2))
    summary = pd.DataFrame(metrics).groupby(['feature_set', 'model']).mean(numeric_only=True)
    summary.to_csv(out / 'validation_summary.csv')
    print(summary.round(4).to_string(), flush=True)
    summary.rmse_log.plot.barh(figsize=(9, 5), title='Mean validation error (log scale; lower is better)')
    plt.tight_layout(); plt.savefig(out / 'validation_comparison.png', dpi=160); plt.close()
    cluster(train, config, out)
    if args.final_test:
        # All choices above use training/validation only.
        test = pd.read_csv(FROZEN / 'test.csv', parse_dates=['date'])
        assert train.date.max() < test.date.min()
        test_metrics, test_preds = [], []
        for key, (model, cols) in fitted.items():
            feature_set, name = key.split('__')
            pred = model.predict(test[cols])
            test_metrics.append(dict(model=name, feature_set=feature_set, **scores(test.y, pred)))
            test_preds.append(export_predictions(test, pred, name, feature_set, 'test_2026'))
        for name in ['persistence', 'climatology']:
            pred = reference(train, test, name)
            test_metrics.append(dict(model=name, feature_set='reference', **scores(test.y, pred)))
            test_preds.append(export_predictions(test, pred, name, 'reference', 'test_2026'))
        pd.DataFrame(test_metrics).to_csv(out / 'test_scores.csv', index=False)
        pd.concat(test_preds).to_csv(out / 'test_predictions.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps(dict(seed=SEED, sklearn=sklearn.__version__,
        final_test=args.final_test, input_hashes=config['file_sha256']), indent=2))


def cluster(train, config, out):
    # Weather profiles, excluding station dummies, target and discharge history.
    cols = config['feature_groups']['weather_same_day'] + config['feature_groups']['rain_history']
    scaler = StandardScaler()
    X = scaler.fit_transform(train[cols])
    rows, models = [], {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        rows.append(dict(k=k, silhouette=silhouette_score(X, model.labels_, sample_size=2000, random_state=SEED), inertia=model.inertia_))
        models[k] = model
    table = pd.DataFrame(rows)
    k = int(table.loc[table.silhouette.idxmax(), 'k'])
    model = models[k]
    table.to_csv(out / 'cluster_selection.csv', index=False)
    assigned = train[['date', 'location'] + cols].copy()
    assigned['cluster'] = model.labels_
    assigned.to_csv(out / 'cluster_assignments.csv', index=False)
    profile = assigned.groupby('cluster')[cols].mean()
    profile['count'] = assigned.groupby('cluster').size()
    profile.to_csv(out / 'cluster_profiles.csv')
    joblib.dump(dict(scaler=scaler, model=model, columns=cols), out / 'clustering.joblib')
    pca = PCA(n_components=2).fit(X)
    xy = pca.transform(X)
    plt.figure(figsize=(8, 5))
    plt.scatter(xy[:, 0], xy[:, 1], c=model.labels_, cmap='tab10', s=5, alpha=.4)
    plt.xlabel('Principal component 1'); plt.ylabel('Principal component 2')
    plt.title(f'Weather groups: k={k}; 2D view explains {pca.explained_variance_ratio_.sum():.0%} of variance')
    plt.tight_layout(); plt.savefig(out / 'clusters.png', dpi=160); plt.close()
    print('Clustering selected k =', k, flush=True)


if __name__ == '__main__':
    main()
