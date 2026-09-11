import importlib
import math
import os
import unittest
from unittest.mock import patch

from src.input.validation import validate_params


class PaperAFieldControlTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            common = importlib.import_module("batch_configs.paper_a_common")
            importlib.reload(common)
            module = importlib.import_module("batch_configs.paper_a_field_control")
            module = importlib.reload(module)
            return module.get_config()

    def test_profile_counts_and_final_statistics(self):
        smoke = self._config("smoke")
        final = self._config("final")

        self.assertEqual(len(smoke["cases"]), 8)
        self.assertEqual(len(final["cases"]), 26)
        self.assertEqual(final["params_default"]["AverageOverRunsN"], 24)
        self.assertEqual(final["params_default"]["nqubits_E"], 16)
        self.assertEqual(final["params_default"]["T"], 60)
        self.assertEqual(final["params_default"]["fragment_sample_count"], 64)
        self.assertTrue(final["params_default"]["store_fragment_samples"])
        self.assertTrue(final["params_default"]["fragment_half_only"])

    def test_cases_are_unique_matched_strength_controls(self):
        config = self._config("final")
        signatures = set()
        for case in config["cases"]:
            validate_params({**config["params_default"], **case})
            signature = (
                case["field_geometry"],
                case["field_profile"],
                case["field_strength_ratio"],
            )
            self.assertNotIn(signature, signatures)
            signatures.add(signature)

            h0 = case["Mironowicz_h0"]
            width = case["Mironowicz_alpha2"]
            self.assertFalse(h0 and width)
            if case["field_profile"] == "uniform":
                self.assertTrue(math.isclose(h0, 0.1 * case["field_strength_ratio"]))
            elif case["field_profile"] == "random":
                self.assertTrue(math.isclose(width, 0.1 * case["field_strength_ratio"]))
            else:
                self.assertEqual(case["field_profile"], "none")
                self.assertEqual(case["field_strength_ratio"], 0.0)
                self.assertEqual(h0, 0.0)
                self.assertEqual(width, 0.0)


if __name__ == "__main__":
    unittest.main()
