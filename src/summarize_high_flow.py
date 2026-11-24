"""Combine completed neural and traditional results only when their datasets agree."""
import json
from pathlib import Path
import pandas as pd
from high_flow_experiment import ROOT


def main():
    root = ROOT / 'results/high_flow'
    paths = [root / 'mlp', root / 'cnn']
    paths += [root / 'classifiers' / f'{features}__{model}'
              for features in ['mlp', 'cnn'] for model in ['logistic', 'tree', 'forest']]
    paths += [root / 'classifiers/persistence']
    reference = json.loads((paths[0] / 'run_info.json').read_text())
    thresholds = pd.read_csv(paths[0] / 'station_thresholds.csv')
    rows = []
    for path in paths:
        info = json.loads((path / 'run_info.json').read_text())
        for key in ['input_sha256', 'splits', 'target']:
            if info[key] != reference[key]:
                raise ValueError(f'Mismatched {key}: {path}; rerun the matching experiment')
        pd.testing.assert_frame_equal(pd.read_csv(path / 'station_thresholds.csv'), thresholds)
        for split in ['val', 'test']:
            keys = ['date', 'target_date', 'location', 'alertTomorrow']
            pd.testing.assert_frame_equal(pd.read_csv(path / f'{split}_predictions.csv')[keys],
                                          pd.read_csv(paths[0] / f'{split}_predictions.csv')[keys])
        rows.append(pd.read_csv(path / 'metrics.csv'))
    results = pd.concat(rows, ignore_index=True)
    results.to_csv(root / 'comparison.csv', index=False)
    print(results[results.split == 'test'].to_string(index=False))


if __name__ == '__main__':
    main()
