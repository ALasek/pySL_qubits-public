import unittest

import numpy as np

from batch_configs.paper_b_mechanism_discrimination import (
    EE_COUPLING,
    FIELD_STRENGTHS,
    FIELD_TO_EE_RATIOS,
    get_config,
)
from src.batch_utils import count_parameter_combinations, iter_parameter_combinations
from src.input.default_params import build_default_params
from src.input.hamiltonian import hamiltonian
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import PAPER_B_MECHANISM_CASES, validate_params


class PaperBMechanismDiscriminationSuiteTests(unittest.TestCase):
    def _params(self, case, nqubits_E=8, field_strength=0.5):
        params = build_default_params()
        params.update(
            {
                "H_SE_Special": "PaperB_staged",
                "H_SE_p": "Const",
                "H_SE_J": 0.1,
                "H_EE_J": EE_COUPLING,
                "Mironowicz_theta": np.pi / 2,
                "Mironowicz_H_E": "Z",
                "nqubits_E": nqubits_E,
                "seed": 1337,
                "PaperB_case": case,
                "PaperB_field_strength": field_strength,
                "PaperB_field_normalization": "exact_rms",
                "evolution_protocol": "staged_write_store",
                "staged_write_time": 1.0,
                "dT": 0.1,
                "printT": 0.1,
                "T": 2.0,
            }
        )
        validate_params(params)
        return params

    def _fields(self, case, nqubits_E=8, field_strength=0.5):
        params = self._params(case, nqubits_E, field_strength)
        return np.asarray(
            hamiltonian(params, seed=params["seed"]).J_E,
            dtype=float,
        )

    def test_config_covers_the_mechanism_design(self):
        config = get_config()
        params = config["params_default"]

        self.assertEqual(config["subdir"], "Paper_B_MechanismDiscrimination_ZX_ExactRMS")
        self.assertEqual(count_parameter_combinations(config["sweep"]), 30)
        self.assertEqual(params["AverageOverRunsN"], 5)
        self.assertEqual(params["H_EE_J"], EE_COUPLING)
        self.assertEqual(params["PaperB_field_normalization"], "exact_rms")
        self.assertEqual(config["sweep"]["nqubits_E"], [8, 10])
        self.assertEqual(config["sweep"]["PaperB_case"], list(PAPER_B_MECHANISM_CASES))
        np.testing.assert_allclose(
            config["sweep"]["PaperB_field_strength"],
            FIELD_STRENGTHS,
        )
        np.testing.assert_allclose(
            np.asarray(FIELD_STRENGTHS) / EE_COUPLING,
            FIELD_TO_EE_RATIOS,
        )
        self.assertEqual(params["dT"], 0.01)
        self.assertEqual(params["T"], 60.0)
        self.assertEqual(params["fragment_sample_count"], 32)
        self.assertEqual(config["analysis"]["setMinTime"], 20.0)
        self.assertEqual(config["analysis"]["metrics"][0], "holevo_discord_fhalf")

        for point in iter_parameter_combinations(params, config["sweep"]):
            validate_params(point)
            seeds = resolve_run_seed_plan_from_params(point)
            self.assertEqual(len(seeds["run_seeds"]), 5)

    def test_quasiperiodic_controls_preserve_the_intended_field_data(self):
        baseline = self._fields("zx_quasiperiodic_chain")
        shuffled = self._fields("zx_quasiperiodic_shuffled_chain")
        sign_scrambled = self._fields("zx_quasiperiodic_signscrambled_chain")

        np.testing.assert_allclose(np.sort(shuffled), np.sort(baseline))
        self.assertFalse(np.allclose(shuffled, baseline))
        np.testing.assert_allclose(np.abs(sign_scrambled), np.abs(baseline))
        self.assertFalse(np.allclose(sign_scrambled, baseline))

        for fields in (baseline, shuffled, sign_scrambled):
            self.assertAlmostEqual(np.sqrt(np.mean(fields**2)), 0.5, places=12)

    def test_two_magnitude_control_has_exact_rms_and_reproducible_levels(self):
        first = self._fields("zx_two_magnitude_chain")
        second = self._fields("zx_two_magnitude_chain")

        np.testing.assert_allclose(second, first)
        self.assertAlmostEqual(np.sqrt(np.mean(first**2)), 0.5, places=12)
        np.testing.assert_allclose(
            np.unique(np.round(np.abs(first), 12)),
            [0.25, np.sqrt(1.75) * 0.5],
        )
        self.assertEqual(np.count_nonzero(first > 0), 4)
        self.assertEqual(np.count_nonzero(first < 0), 4)


if __name__ == "__main__":
    unittest.main()
