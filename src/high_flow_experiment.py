"""Shared next-day high-flow experiment for Louis's networks and matched classifiers.

Ai was used to clarify the structure: High flow means above a station's training-only 95th percentile, not verified flooding.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/nepal_flood_clean.csv'
MLP_FEATURES = ['precipitation_mm', 'rain1', 'rain2', 'rain3d', 'rain7d',
                'soil_moisture_0_100cm_m3m3', 'precipitation_hours',
                'relative_humidity_mean_pct', 'monthSin', 'monthCos',
                'ratioToday', 'ratioYesterday']
CNN_FEATURES = [c for lag in [3, 2, 1] for c in
                [f'rain_{lag}', f'hours_{lag}', f'soil_{lag}', f'ratio_{lag}']]
CNN_FEATURES += ['precipitation_mm', 'precipitation_hours', 'soil_moisture_0_100cm_m3m3', 'ratioToday']


def prepare_data(path=DATA):
    frame = pd.read_csv(path, parse_dates=['date']).sort_values(['location', 'date']).reset_index(drop=True)
    if frame.duplicated(['location', 'date']).any():
        raise ValueError('Duplicate station-days')
    by = frame.groupby('location')
    if not by.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Daily continuity is required for next-day labels and lag features')
    frame['target_date'] = by.date.shift(-1)
    frame['dischargeTomorrow'] = by.river_discharge_m3s.shift(-1)
    for lag in [1, 2, 3]:
        for prefix, column in [('rain', 'precipitation_mm'), ('hours', 'precipitation_hours'),
                               ('soil', 'soil_moisture_0_100cm_m3m3'), ('discharge', 'river_discharge_m3s')]:
            frame[f'{prefix}_{lag}'] = by[column].shift(lag)
    frame['rain1'], frame['rain2'] = frame.rain_1, frame.rain_2
    for n in [3, 7]:
        frame[f'rain{n}d'] = by.precipitation_mm.transform(lambda s: s.rolling(n).sum())
    frame['dischargeYesterday'] = frame.discharge_1
    frame['monthSin'] = np.sin(2 * np.pi * frame.date.dt.month / 12)
    frame['monthCos'] = np.cos(2 * np.pi * frame.date.dt.month / 12)
    frame = frame.dropna(subset=['rain7d', 'discharge_1', 'dischargeTomorrow']).sort_values(['date', 'location']).reset_index(drop=True)

    # Retain Louis's date boundaries, but purge rows whose tomorrow label reaches
    # the next evaluation period. Keep all stations on a date in the same split.
    test_start = frame.loc[frame.date >= frame.date.quantile(.80), 'date'].min()
    full_train = frame[frame.date < test_start]
    val_start = full_train.iloc[int(len(full_train) * .80)].date
    train = frame[(frame.date < val_start) & (frame.target_date < val_start)].copy()
    val = frame[(frame.date >= val_start) & (frame.date < test_start) & (frame.target_date < test_start)].copy()
    test = frame[frame.date >= test_start].copy()
    thresholds = train.groupby('location').dischargeTomorrow.quantile(.95)
    if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError('Station thresholds must be finite and positive')
    splits = dict(train=train, val=val, test=test)
    for name, part in splits.items():
        part['threshold'] = part.location.map(thresholds)
        part['alertTomorrow'] = (part.dischargeTomorrow > part.threshold).astype(int)
        part['ratioToday'] = part.river_discharge_m3s / part.threshold
        part['ratioYesterday'] = part.dischargeYesterday / part.threshold
        for lag in [1, 2, 3]:
            part[f'ratio_{lag}'] = part[f'discharge_{lag}'] / part.threshold
        if not np.isfinite(part[MLP_FEATURES + CNN_FEATURES + ['threshold']]).all().all():
            raise ValueError(f'Non-finite features in {name}')
        if part.alertTomorrow.nunique() != 2:
            raise ValueError(f'{name} must contain both classes for this evaluation')
    assert train.target_date.max() < val.date.min()
    assert val.target_date.max() < test.date.min()
    return splits, thresholds


def arrays(splits, feature_set):
    features = MLP_FEATURES if feature_set == 'mlp' else CNN_FEATURES
    scaler = StandardScaler().fit(splits['train'][features])
    xs = {k: scaler.transform(v[features]) for k, v in splits.items()}
    return xs, {k: v.alertTomorrow.to_numpy() for k, v in splits.items()}, scaler, features


def select_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1)])


def save_results(out, splits, thresholds, scores, decision_threshold, model_name, features, extra=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ['val', 'test']:
        part = splits[split]
        score = np.asarray(scores[split])
        pred = score >= decision_threshold
        y = part.alertTomorrow.to_numpy()
        onset = (y == 1) & (part.ratioToday.to_numpy() <= 1)
        rows.append(dict(model=model_name, split=split, threshold=decision_threshold,
            average_precision=average_precision_score(y, score), prevalence=float(y.mean()),
            precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0), onsets=int(onset.sum()),
            onsets_caught=int(pred[onset].sum()), onset_recall=float(pred[onset].mean()) if onset.any() else None))
        predictions = part[['date', 'target_date', 'location', 'dischargeTomorrow', 'threshold', 'alertTomorrow']].copy()
        predictions['score'] = score
        predictions['predicted_high_flow'] = pred.astype(int)
        predictions.to_csv(out / f'{split}_predictions.csv', index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'metrics.csv', index=False)
    thresholds.rename('high_flow_threshold_m3s').to_csv(out / 'station_thresholds.csv')
    metadata = dict(model=model_name, seed=42, target='next-day training-station 95th-percentile high flow (not confirmed flood)',
        features=features, input_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        splits={k: dict(rows=len(v), feature_start=str(v.date.min().date()), feature_end=str(v.date.max().date()),
                       target_start=str(v.target_date.min().date()), target_end=str(v.target_date.max().date()),
                       positives=int(v.alertTomorrow.sum())) for k,v in splits.items()},
        **(extra or {}))
    (out / 'run_info.json').write_text(json.dumps(metadata, indent=2))
    return metrics
