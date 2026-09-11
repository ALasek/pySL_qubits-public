import unittest

import numpy as np

from src.export.run_store import StoredRunResult
from src.input.wavefunction import wavefunction


class _MetricOnly:
    I_S_Ef_fractionsT_runAv = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.1, 0.2],
            [0.8, 0.2, 0.9],
            [1.0, 0.3, 1.0],
        ]
    )

    def redundancy_fractionScore(self, cutoff=0.5, minfraction=0.4, minTime=0):
        raise AssertionError("best-time path should not be used")


class FixedTimeRedundancyTests(unittest.TestCase):
    def test_live_result_uses_only_requested_time(self):
        result = wavefunction.redundancy_fractionScore_at_T(_MetricOnly(), 0)

        self.assertEqual(result[1], 0)

    def test_stored_result_uses_only_requested_time(self):
        result = StoredRunResult.redundancy_fractionScore_at_T(_MetricOnly(), 0)

        self.assertEqual(result[1], 0)

    def test_requested_time_before_minimum_is_rejected(self):
        result = StoredRunResult.redundancy_fractionScore_at_T(_MetricOnly(), 0, minTime=1)

        self.assertEqual(result, [1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
