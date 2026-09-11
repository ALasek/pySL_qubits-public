import importlib
import math
import os
import unittest
from unittest.mock import patch


class PaperASubmissionTests(unittest.TestCase):
    def _config(self, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            common = importlib.import_module("batch_configs.paper_a_common")
            importlib.reload(common)
            module = importlib.import_module("batch_configs.paper_a_submission_lambda")
            module = importlib.reload(module)
            return module, module.get_config()

    def test_smoke_and_final_case_counts(self):
        _, smoke = self._config("smoke")
        _, final = self._config("final")

        self.assertEqual(len(smoke["cases"]), 8)
        self.assertEqual(smoke["params_default"]["AverageOverRunsN"], 1)
        self.assertEqual(len(final["cases"]), 38)
        self.assertEqual(final["params_default"]["AverageOverRunsN"], 24)
        self.assertEqual(final["params_default"]["nqubits_E"], 16)

    def test_every_case_matches_declared_lambda(self):
        module, config = self._config("final")

        for case in config["cases"]:
            value = module.alignment_lambda(case["psi_bias"], case["Mironowicz_theta"])
            self.assertTrue(math.isclose(value, case["lambda_target"], abs_tol=1e-12))

        pairs = {(case["psi_bias"], case["Mironowicz_theta"]) for case in config["cases"]}
        self.assertEqual(len(pairs), len(config["cases"]))


if __name__ == "__main__":
    unittest.main()
