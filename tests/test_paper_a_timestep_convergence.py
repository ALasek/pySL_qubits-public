import importlib
import math
import os
import unittest
from unittest.mock import patch

from src.input.validation import validate_params


class PaperATimestepConvergenceTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            module = importlib.import_module("batch_configs.paper_a_timestep_convergence")
            module = importlib.reload(module)
            return module.get_config()

    def test_final_profile_matches_production_except_for_half_timestep(self):
        config = self._config("final")
        defaults = config["params_default"]

        self.assertEqual(config["subdir"], "Paper_A_final_TimestepConvergence")
        self.assertEqual(len(config["cases"]), 3)
        self.assertEqual(defaults["AverageOverRunsN"], 4)
        self.assertEqual(defaults["nqubits_E"], 16)
        self.assertEqual(defaults["fragment_sample_count"], 64)
        self.assertTrue(math.isclose(defaults["dT"], 0.001))
        self.assertEqual(config["cases"][0]["T"], 40)

        for case in config["cases"]:
            params = {**defaults, **case}
            self.assertTrue(math.isclose(params["reference_dT"], 0.002))
            validate_params(params)

    def test_smoke_profile_preserves_all_three_regimes(self):
        smoke = self._config("smoke")
        final = self._config("final")

        self.assertEqual(
            [case["convergence_case"] for case in smoke["cases"]],
            [case["convergence_case"] for case in final["cases"]],
        )
        self.assertEqual(smoke["params_default"]["AverageOverRunsN"], 1)
        self.assertEqual(smoke["params_default"]["nqubits_E"], 6)
        self.assertTrue(math.isclose(smoke["params_default"]["dT"], 0.005))


if __name__ == "__main__":
    unittest.main()
