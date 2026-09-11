import importlib
import os
import unittest
from unittest.mock import patch

from src.batch_utils import compute_stored_run_id
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import validate_params


class PaperALowFieldRefinementTests(unittest.TestCase):
    def _config(self, module, profile):
        with patch.dict(os.environ, {"PYSL_PAPER_A_PROFILE": profile}):
            return importlib.reload(importlib.import_module(module)).get_config()

    def test_final_preserves_reference_contract_except_cutoff(self):
        reference = self._config("batch_configs.paper_a_field_control", "final")
        config = self._config("batch_configs.paper_a_low_field_refinement", "final")
        expected = {**reference["params_default"], "clampISE": 0.0}
        self.assertEqual(config["params_default"], expected)
        self.assertEqual(config["subdir"], "Paper_A_final_LowFieldRefinement")
        seeds = resolve_run_seed_plan_from_params(config["params_default"])
        self.assertEqual(seeds, resolve_run_seed_plan_from_params(reference["params_default"]))
        self.assertEqual(len(seeds["run_seeds"]), 24)
        self.assertEqual(config["analysis"]["late_window"], [30.0, 60.0])

    def test_profiles_cover_eight_unique_aligned_cases(self):
        for profile in ("smoke", "final"):
            config = self._config("batch_configs.paper_a_low_field_refinement", profile)
            expected = {(ratio, field) for ratio in (0.05, 0.10, 0.15, 0.20)
                        for field in ("uniform", "random")}
            self.assertEqual({(c["field_strength_ratio"], c["field_profile"])
                              for c in config["cases"]}, expected)
            run_ids = set()
            for case in config["cases"]:
                params = {**config["params_default"], **case}
                validate_params(params)
                self.assertEqual(params["psi_bias"], 0.5)
                self.assertEqual(params["Mironowicz_theta"], 0.0)
                self.assertAlmostEqual(params["Mironowicz_h0"] + params["Mironowicz_alpha2"],
                                       0.1 * case["field_strength_ratio"])
                self.assertEqual(params["Mironowicz_h0"] * params["Mironowicz_alpha2"], 0)
                run_ids.add(compute_stored_run_id(params))
            self.assertEqual(len(run_ids), 8)


if __name__ == "__main__":
    unittest.main()
