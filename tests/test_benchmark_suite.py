import json
import os
import tempfile
import unittest

import numpy as np

from benchmark_suite import _replace_artifact_dir, _write_json, summarize_results


class BenchmarkSuiteTests(unittest.TestCase):
    def test_result_summary_uses_magnitude_for_complex_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.npz")
            np.savez_compressed(path, correlator=np.array([3 + 4j, 0 + 0j]))

            summary = summarize_results(path, ["correlator"])["correlator"]

        self.assertTrue(summary["complex_magnitude"])
        self.assertEqual(summary["max"], 5.0)
        self.assertEqual(summary["final"], 0.0)

    def test_nonfinite_final_summary_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = os.path.join(tmp, "results.npz")
            summary_path = os.path.join(tmp, "summary.json")
            np.savez_compressed(results_path, metric=np.array([1.0, np.nan]))

            summary = summarize_results(results_path, ["metric"])
            _write_json(summary_path, summary)
            with open(summary_path, encoding="utf-8") as f:
                loaded = json.load(f)

        self.assertIsNone(loaded["metric"]["final"])

    def test_baseline_target_must_stay_under_baseline_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            baseline_root = os.path.join(tmp, "benchmarks", "baselines")
            outside = os.path.join(tmp, "outside", "case")
            os.makedirs(source)

            with self.assertRaises(ValueError):
                _replace_artifact_dir(source, outside, baseline_root, force=False)


if __name__ == "__main__":
    unittest.main()
