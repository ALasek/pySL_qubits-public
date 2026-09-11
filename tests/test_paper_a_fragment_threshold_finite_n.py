import math
import unittest
from unittest.mock import patch

from scripts import run_paper_a_fragment_threshold_finite_n as runner
from src.batch_utils import iter_parameter_sets
from src.input.validation import validate_params


class PaperAFragmentThresholdFiniteNTests(unittest.TestCase):
    def test_final_contract(self):
        config = runner.load_config("final")

        self.assertEqual(len(config["cases"]), 4)
        self.assertEqual(config["params_default"]["AverageOverRunsN"], 8)
        self.assertEqual(config["params_default"]["fragment_sample_count"], 64)
        self.assertEqual(config["params_default"]["T"], 36)
        self.assertEqual(config["params_default"]["printT"], 36)
        self.assertEqual({case["nqubits_E"] for case in config["cases"]}, {20, 24})
        self.assertEqual({case["lambda_target"] for case in config["cases"]}, {0.75, 1.0})

        for params in iter_parameter_sets(config["params_default"], cases=config["cases"]):
            validate_params(params)
            lambda_value = math.sin(float(params["Mironowicz_theta"])) ** 2
            self.assertAlmostEqual(lambda_value, float(params["lambda_target"]))
            self.assertEqual(params["H_EE_J"], 0.0)
            self.assertEqual(params["Mironowicz_alpha2"], 0.0)
            self.assertTrue(params["store_fragment_samples"])

    def test_smoke_allocates_largest_environment(self):
        config = runner.load_config("smoke")

        self.assertEqual(len(config["cases"]), 1)
        self.assertEqual(config["cases"][0]["nqubits_E"], 24)
        self.assertEqual(config["params_default"]["AverageOverRunsN"], 1)
        self.assertEqual(config["params_default"]["T"], 0.02)

    def test_n26_profiles_preserve_physics_and_original_batch(self):
        original = runner.load_config("final")
        for profile in ("smoke", "final"):
            config = runner.load_config(profile, n26=True)
            params = config["params_default"]
            self.assertEqual(config["subdir"], f"Paper_A_{profile}_FragmentThresholdN26")
            self.assertEqual(len(config["cases"]), 2)
            self.assertEqual(params["nqubits_E"], 26)
            self.assertEqual(params["dT"], 0.002)
            self.assertTrue(params["log_gpu_memory"])
            self.assertTrue(params["compute_discord"])
            self.assertTrue(params["fragment_half_only"])
            self.assertEqual(params["fragment_sample_count"], 64 if profile == "final" else 4)
            self.assertEqual(params["AverageOverRunsN"], 8 if profile == "final" else 1)
            self.assertEqual(params["T"], params["printT"])
            self.assertEqual(params["T"], 36 if profile == "final" else 0.02)
            for point in iter_parameter_sets(params, cases=config["cases"]):
                validate_params(point)
                self.assertEqual(point["nqubits_E"], 26)
                self.assertAlmostEqual(math.sin(point["Mironowicz_theta"]) ** 2, point["lambda_target"])
                self.assertEqual(point["psi_bias"], 0.5)
                for key in ("H_SE_Special", "H_SE_J", "H_EE_J", "seed", "Mironowicz_alpha2", "Mironowicz_h0"):
                    self.assertEqual(point[key], original["params_default"][key])

        restored = runner.load_config("final")
        self.assertEqual(restored["cases"], original["cases"])
        self.assertEqual(restored["params_default"], original["params_default"])
        self.assertNotEqual(restored["subdir"], config["subdir"])

    def test_n26_dry_run_does_not_launch_simulations(self):
        with patch.object(runner.subprocess, "run") as run:
            self.assertEqual(runner.main(["--n26", "--profile", "final", "--dry-run"]), 0)
        run.assert_not_called()

    def test_n26_dispatches_separate_config(self):
        with patch.object(runner.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(runner.main(["--n26", "--profile", "smoke", "--manifest-only"]), 0)
        command = run.call_args.args[0]
        self.assertIn(runner.CONFIG_N26, command)
        self.assertNotIn(runner.CONFIG, command)
        self.assertIn("--manifest-only", command)
        self.assertNotIn("--skip-existing", command)
        self.assertEqual(run.call_args.kwargs["env"][runner.PROFILE_ENV], "smoke")


if __name__ == "__main__":
    unittest.main()
