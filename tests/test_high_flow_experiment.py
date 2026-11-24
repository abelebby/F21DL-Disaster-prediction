"""Checks for split leakage and alignment in the shared classification experiment."""
import unittest
import numpy as np
import pandas as pd
from src.high_flow_experiment import DATA, prepare_data, arrays, select_threshold


class HighFlowExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits, cls.thresholds = prepare_data()

    def test_target_dates_precede_next_period(self):
        for before, after in [('train', 'val'), ('val', 'test')]:
            self.assertLess(self.splits[before].target_date.max(), self.splits[after].date.min())
        for part in self.splits.values():
            self.assertTrue((part.target_date - part.date).eq(pd.Timedelta(days=1)).all())

    def test_targets_match_actual_next_day_measurements(self):
        raw = pd.read_csv(DATA, parse_dates=['date']).set_index(['location', 'date'])
        for part in self.splits.values():
            keys = pd.MultiIndex.from_arrays([part.location, part.target_date])
            np.testing.assert_allclose(part.dischargeTomorrow, raw.river_discharge_m3s.reindex(keys))

    def test_thresholds_and_scaling_use_training_only(self):
        expected = self.splits['train'].groupby('location').dischargeTomorrow.quantile(.95)
        pd.testing.assert_series_equal(self.thresholds, expected)
        for feature_set in ['mlp', 'cnn']:
            xs, ys, scaler, features = arrays(self.splits, feature_set)
            np.testing.assert_allclose(scaler.mean_, self.splits['train'][features].mean())
            for name, part in self.splits.items():
                self.assertEqual(len(xs[name]), len(part))
                np.testing.assert_array_equal(ys[name], part.alertTomorrow)
                self.assertTrue(np.isfinite(xs[name]).all())

    def test_cutoff_selected_from_supplied_validation_scores(self):
        self.assertAlmostEqual(select_threshold([0, 0, 1, 1], [.1, .2, .7, .9]), .7)


if __name__ == '__main__':
    unittest.main()
