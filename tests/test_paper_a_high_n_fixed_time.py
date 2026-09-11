import importlib
import math
import os
import unittest
from unittest.mock import patch

from src.input.validation import validate_params


class PaperAHighNFixedTimeTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            module = importlib.import_module("batch_configs.paper_a_high_n_fixed_time")
            module = importlib.reload(module)
            return module.get_config()

    def test_final_profile_is_one_n24_fixed_time_case(self):
        config = self._config("final")
        params = {**config["params_default"], **config["cases"][0]}

        self.assertEqual(config["subdir"], "Paper_A_final_N24FixedTimeValidation")
        self.assertEqual(len(config["cases"]), 1)
        self.assertEqual(params["nqubits_E"], 24)
        self.assertEqual(params["AverageOverRunsN"], 8)
        self.assertEqual(params["fragment_sample_count"], 32)
        self.assertEqual(params["T"], 36)
        self.assertEqual(params["printT"], 36)
        self.assertTrue(math.isclose(params["dT"], 0.002))
        self.assertTrue(math.isclose(params["Mironowicz_theta"], math.pi / 4.0))
        validate_params(params)

    def test_smoke_profile_allocates_the_final_hilbert_space(self):
        config = self._config("smoke")
        params = {**config["params_default"], **config["cases"][0]}

        self.assertEqual(params["nqubits_E"], 24)
        self.assertEqual(params["AverageOverRunsN"], 1)
        self.assertEqual(params["fragment_sample_count"], 4)
        self.assertTrue(math.isclose(params["T"], 0.02))
        self.assertTrue(math.isclose(params["printT"], 0.02))
        validate_params(params)


if __name__ == "__main__":
    unittest.main()
