"""Traditional classifiers matched to Louis's revised next-day high-flow setup."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from high_flow_experiment import ROOT, prepare_data, arrays, select_threshold, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/high_flow/classifiers')
    args = parser.parse_args()
    splits, thresholds = prepare_data()
    rows = []
    for feature_set in ['mlp', 'cnn']:
        xs, ys, scaler, features = arrays(splits, feature_set)
        # Fixed settings, not selected by test performance. Only the probability
        # cutoff is tuned on validation F1, matching the networks' procedure.
        models = dict(logistic=LogisticRegression(max_iter=2000, random_state=42),
                      tree=DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
                      forest=RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1))
        for name, model in models.items():
            model.fit(xs['train'], ys['train'])
            scores = {k: model.predict_proba(xs[k])[:, 1] for k in ['val', 'test']}
            cutoff = select_threshold(ys['val'], scores['val'])
            label = f'{feature_set}__{name}'
            metrics = save_results(args.results_dir / label, splits, thresholds, scores, cutoff, label, features,
                                   dict(parameters=model.get_params(), feature_set=feature_set))
            rows.append(metrics)
            print(metrics.to_string(index=False), flush=True)
    scores = {k: v.ratioToday.to_numpy() for k,v in splits.items() if k != 'train'}
    rows.append(save_results(args.results_dir / 'persistence', splits, thresholds, scores,
                            float(np.nextafter(1., float('inf'))), 'persistence', ['ratioToday'],
                            dict(score_description='today discharge / station threshold; not a probability')))
    pd.concat(rows).to_csv(args.results_dir / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()
