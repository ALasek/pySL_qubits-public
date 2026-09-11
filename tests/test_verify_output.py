import json
import os
import tempfile
import unittest

import numpy as np

from verify_output import _manifest_cases, compare_arrays_detailed


class VerifyOutputTests(unittest.TestCase):
    def test_comparison_rejects_nan_position_mismatch(self):
        detail = compare_arrays_detailed(
            "metric",
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
            atol=1e-6,
        )

        self.assertEqual(detail["status"], "fail")
        self.assertEqual(detail["nan_mismatches"], 1)

    def test_comparison_uses_combined_absolute_and_relative_tolerance(self):
        detail = compare_arrays_detailed(
            "metric",
            np.array([100.01, 1e-7]),
            np.array([100.0, 0.0]),
            atol=1e-6,
            rtol=2e-4,
        )

        self.assertEqual(detail["status"], "ok")

    def test_manifest_case_paths_are_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "cases": [
                            {"id": "case-a", "reference": "baselines/case-a"},
                            {"id": "case-b", "reference": "baselines/case-b"},
                        ]
                    },
                    f,
                )

            _manifest, cases = _manifest_cases(manifest_path, selected=["case-b"])

        self.assertEqual(cases[0][0]["id"], "case-b")
        self.assertEqual(cases[0][1], os.path.join(tmp, "baselines", "case-b"))


if __name__ == "__main__":
    unittest.main()
