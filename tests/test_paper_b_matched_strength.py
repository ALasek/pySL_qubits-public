import unittest

import numpy as np

from batch_configs.paper_b_matched_strength import (
    EE_COUPLING,
    FIELD_STRENGTHS,
    FIELD_TO_EE_RATIOS,
    get_config,
)
from src.batch_utils import count_parameter_combinations, iter_parameter_combinations
from src.input.default_params import build_default_params
from src.input.hamiltonian import hamiltonian
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import PAPER_B_MATCHED_CASES, validate_params


class PaperBMatchedStrengthSuiteTests(unittest.TestCase):
    def test_config_covers_the_matched_strength_design(self):
        config = get_config()
        params = config["params_default"]

        self.assertEqual(config["subdir"], "Paper_B_MatchedStrength_ZX_ExactRMS")
        self.assertEqual(count_parameter_combinations(config["sweep"]), 24)
        self.assertEqual(params["AverageOverRunsN"], 5)
        self.assertEqual(params["H_EE_J"], EE_COUPLING)
        self.assertEqual(params["PaperB_field_normalization"], "exact_rms")
        self.assertEqual(config["sweep"]["nqubits_E"], [8, 10])
        self.assertEqual(config["sweep"]["PaperB_case"], list(PAPER_B_MATCHED_CASES))
        np.testing.assert_allclose(
            config["sweep"]["PaperB_field_strength"],
            FIELD_STRENGTHS,
        )
        self.assertEqual(params["dT"], 0.01)
        self.assertEqual(params["T"], 60.0)
        self.assertEqual(params["printT"], 2.0)
        self.assertEqual(params["fragment_sample_count"], 32)
        self.assertEqual(config["analysis"]["setMinTime"], 20.0)
        self.assertEqual(config["analysis"]["metrics"][0], "holevo_discord_fhalf")

        for point in iter_parameter_combinations(params, config["sweep"]):
            validate_params(point)
            seeds = resolve_run_seed_plan_from_params(point)
            self.assertEqual(len(seeds["run_seeds"]), 5)
            self.assertEqual(seeds["seed_strategy"], "SeedSequence.spawn")

    def test_each_field_pattern_has_exact_active_site_rms_and_is_reproducible(self):
        for case in PAPER_B_MATCHED_CASES:
            params = build_default_params()
            params.update(
                {
                    "H_SE_Special": "PaperB_staged",
                    "H_SE_p": "Const",
                    "H_SE_J": 0.1,
                    "H_EE_J": EE_COUPLING,
                    "Mironowicz_theta": np.pi / 2,
                    "Mironowicz_H_E": "Z",
                    "nqubits_E": 8,
                    "seed": 1337,
                    "PaperB_case": case,
                    "PaperB_field_strength": 0.5,
                    "PaperB_field_normalization": "exact_rms",
                    "evolution_protocol": "staged_write_store",
                    "staged_write_time": 1.0,
                    "dT": 0.1,
                    "printT": 0.1,
                    "T": 2.0,
                }
            )
            validate_params(params)
            first = hamiltonian(params, seed=params["seed"])
            second = hamiltonian(params, seed=params["seed"])
            fields = np.asarray(first.J_E, dtype=float)

            self.assertAlmostEqual(np.sqrt(np.mean(fields**2)), 0.5, places=12)
            np.testing.assert_allclose(fields, np.asarray(second.J_E, dtype=float))
            if case == "zx_binary_chain":
                np.testing.assert_allclose(np.abs(fields), 0.5)
                self.assertAlmostEqual(np.mean(fields), 0.0, places=12)
            elif case == "zx_uniform_chain":
                np.testing.assert_allclose(fields, 0.5)

    def test_invalid_field_normalization_is_rejected(self):
        params = build_default_params()
        params.update(
            {
                "H_SE_Special": "PaperB_staged",
                "PaperB_case": "zx_gaussian_chain",
                "PaperB_field_strength": 0.5,
                "PaperB_field_normalization": "not-a-normalization",
                "evolution_protocol": "staged_write_store",
                "staged_write_time": 1.0,
                "dT": 0.1,
                "printT": 0.1,
                "T": 2.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "PaperB_field_normalization"):
            validate_params(params)

    def test_exact_rms_rejects_an_empty_active_site_mask(self):
        params = build_default_params()
        params.update(
            {
                "H_SE_Special": "PaperB_staged",
                "H_SE_p": "Const",
                "H_SE_J": 0.1,
                "H_EE_J": EE_COUPLING,
                "Mironowicz_theta": np.pi / 2,
                "Mironowicz_H_E": "Z",
                "nqubits_E": 1,
                "PaperB_case": "zx_gaussian_disjoint_sources",
                "PaperB_field_strength": 0.5,
                "PaperB_field_normalization": "exact_rms",
                "evolution_protocol": "staged_write_store",
                "staged_write_time": 1.0,
                "T": 2.0,
            }
        )

        with self.assertRaisesRegex(ValueError, "without active sites"):
            hamiltonian(params, seed=1337)


if __name__ == "__main__":
    unittest.main()
