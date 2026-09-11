import importlib
import math
import os
import unittest
from unittest.mock import patch

from src.input.validation import validate_params


class PaperAHigherNValidationTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_HIGH_N_PROFILE": profile}):
            module = importlib.import_module("batch_configs.paper_a_higher_n_validation")
            module = importlib.reload(module)
            return module.get_config()

    def test_profiles_keep_the_same_focused_cases(self):
        smoke = self._config("smoke")
        final = self._config("final")

        self.assertEqual(len(smoke["cases"]), 5)
        self.assertEqual(smoke["cases"], final["cases"])
        self.assertEqual(final["params_default"]["nqubits_E"], 20)
        self.assertEqual(final["params_default"]["AverageOverRunsN"], 8)
        self.assertEqual(final["params_default"]["fragment_sample_count"], 32)
        self.assertEqual(final["params_default"]["T"], 60)

    def test_cases_are_valid_and_have_one_field_profile(self):
        config = self._config("final")
        labels = set()
        for case in config["cases"]:
            validate_params({**config["params_default"], **case})
            self.assertNotIn(case["validation_case"], labels)
            labels.add(case["validation_case"])
            self.assertFalse(case["Mironowicz_h0"] and case["Mironowicz_alpha2"])

        no_field = config["cases"][0]
        self.assertTrue(math.isclose(no_field["Mironowicz_theta"], math.pi / 4.0))
        self.assertEqual(no_field["psi_bias"], 0.5)


if __name__ == "__main__":
    unittest.main()
