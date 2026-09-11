import importlib
import math
import os
import unittest
from unittest.mock import patch

from src.input.validation import validate_params


class PaperAHolevoPlateauFullFragmentsTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            module = importlib.import_module("batch_configs.paper_a_holevo_plateau_full_fragments")
            module = importlib.reload(module)
            return module.get_config()

    def test_final_profile_is_one_direct_full_fragment_run(self):
        config = self._config("final")
        params = {**config["params_default"], **config["cases"][0]}

        self.assertEqual(config["subdir"], "Paper_A_final_HolevoPlateauFullFragments")
        self.assertEqual(len(config["cases"]), 1)
        self.assertFalse(params["fragment_half_only"])
        self.assertEqual(params["fragment_sample_count"], 64)
        self.assertEqual(params["AverageOverRunsN"], 24)
        self.assertEqual(params["nqubits_E"], 16)
        self.assertEqual(params["psi_bias"], 0.5)
        self.assertTrue(math.isclose(params["Mironowicz_theta"], math.pi / 3.0))
        validate_params(params)

    def test_smoke_profile_preserves_full_fragment_path(self):
        config = self._config("smoke")
        params = {**config["params_default"], **config["cases"][0]}

        self.assertFalse(params["fragment_half_only"])
        self.assertEqual(params["fragment_sample_count"], 4)
        self.assertEqual(params["AverageOverRunsN"], 1)
        self.assertEqual(params["nqubits_E"], 6)
        validate_params(params)


if __name__ == "__main__":
    unittest.main()
