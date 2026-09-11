import unittest

import numpy as np

from src.analysis.correlator_metrics import connected_correlator_metrics


class CorrelatorMetricTests(unittest.TestCase):
    def test_connected_correlator_spread_metrics(self):
        C = np.zeros((1, 3, 3), dtype=np.complex64)
        C[0, 1, 1] = 2.0
        C[0, 0, 1] = 1.0
        C[0, 2, 0] = 1.0j

        metrics = connected_correlator_metrics(C)

        self.assertAlmostEqual(metrics["corr_total_connected_weight"][0], 6.0)
        self.assertAlmostEqual(metrics["corr_offdiag_fraction"][0], 2.0 / 6.0)
        self.assertAlmostEqual(metrics["corr_nn_weight"][0], 1.0)
        self.assertAlmostEqual(metrics["corr_nn_fraction"][0], 0.5)
        self.assertAlmostEqual(metrics["corr_spread_radius"][0], np.sqrt(5.0 / 6.0))

    def test_zero_connected_weight_has_zero_spread(self):
        metrics = connected_correlator_metrics(np.zeros((2, 2, 2)))

        np.testing.assert_allclose(metrics["corr_spread_radius"], [0.0, 0.0])
        np.testing.assert_allclose(metrics["corr_offdiag_fraction"], [0.0, 0.0])
        np.testing.assert_allclose(metrics["corr_nn_fraction"], [0.0, 0.0])

    def test_unavailable_pre_reference_rows_remain_nan(self):
        C = np.zeros((2, 2, 1), dtype=np.complex64)
        C[0] = np.nan

        metrics = connected_correlator_metrics(C, source_indices=[0])

        for values in metrics.values():
            self.assertTrue(np.isnan(values[0]))
        self.assertEqual(metrics["corr_spread_radius"][1], 0.0)

    def test_rectangular_sources_use_physical_source_positions(self):
        C = np.zeros((1, 4, 1), dtype=np.complex64)
        C[0, 3, 0] = 1j

        metrics = connected_correlator_metrics(C, source_indices=[1])

        self.assertAlmostEqual(metrics["corr_spread_radius"][0], 2.0)
        self.assertAlmostEqual(metrics["corr_commutator_spread_radius"][0], 2.0)
        self.assertAlmostEqual(metrics["corr_total_commutator_weight"][0], 4.0)

    def test_rectangular_correlator_requires_source_indices(self):
        with self.assertRaisesRegex(ValueError, "source_indices"):
            connected_correlator_metrics(np.zeros((1, 3, 1)))


if __name__ == "__main__":
    unittest.main()
