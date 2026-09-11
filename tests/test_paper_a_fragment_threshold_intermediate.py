import copy
import math
import unittest
from unittest.mock import patch

from scripts import run_paper_a_fragment_threshold_finite_n as runner
from src.batch_utils import iter_parameter_sets
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import validate_params


class PaperAIntermediateThresholdTests(unittest.TestCase):
    def test_final_contract(self):
        config = runner.load_config("final", intermediate=True)
        params = config["params_default"]
        self.assertEqual(config["subdir"], "Paper_A_final_FragmentThresholdIntermediateN")
        self.assertEqual(len(config["cases"]), 6)
        self.assertEqual({case["nqubits_E"] for case in config["cases"]}, {10, 14})
        self.assertEqual(params["AverageOverRunsN"], 12)
        self.assertEqual(params["fragment_sample_count"], 32)
        self.assertEqual(params["fragment_quantiles"], [0.1, 0.5])
        self.assertEqual(params["T"], 36)
        self.assertEqual(params["printT"], 36)
        self.assertEqual(params["dT"], 0.002)
        self.assertEqual(params["seed"], 1337)
        self.assertEqual(resolve_run_seed_plan_from_params(params)["run_seeds"], [
            3923535749, 3212275105, 4130349263, 1365772060,
            2147774961, 3688406951, 1953186726, 938346551,
            1860405821, 323417900, 1840093682, 2427933164,
        ])
        for point in iter_parameter_sets(params, cases=config["cases"]):
            validate_params(point)
            p, theta = point["psi_bias"], point["Mironowicz_theta"]
            dot = 2 * math.sqrt(p * (1 - p)) * math.cos(theta) + (2 * p - 1) * math.sin(theta)
            self.assertAlmostEqual(1 - dot**2, point["lambda_target"])
            self.assertEqual(point["H_SE_Special"], "Mironowicz_rand")
            self.assertEqual(point["H_SE_J"], 0.1)
            self.assertEqual(point["H_EE_J"], 0.0)
            self.assertEqual(point["Mironowicz_alpha2"], 0.0)
            self.assertEqual(point["Mironowicz_h0"], 0.0)
            for key in ("compute_discord", "store_fragment_samples", "fragment_reuse_samples", "fragment_half_only"):
                self.assertTrue(point[key])
            for key in ("store_psi", "store_correlators", "save_legacy_rundata"):
                self.assertFalse(point[key])

    def test_smoke_covers_every_case_without_changing_production(self):
        production = copy.deepcopy(runner.load_config("final", intermediate=True))
        smoke = runner.load_config("smoke", intermediate=True)
        self.assertEqual(smoke["cases"], production["cases"])
        self.assertEqual(smoke["params_default"]["AverageOverRunsN"], 1)
        self.assertEqual(smoke["params_default"]["fragment_sample_count"], 4)
        self.assertEqual(smoke["params_default"]["T"], 0.02)
        self.assertEqual(smoke["params_default"]["printT"], 0.02)
        for point in iter_parameter_sets(smoke["params_default"], cases=smoke["cases"]):
            validate_params(point)
        self.assertEqual(runner.load_config("final", intermediate=True), production)

    def test_existing_batches_are_unchanged(self):
        original = copy.deepcopy(runner.load_config("final"))
        n26 = copy.deepcopy(runner.load_config("final", n26=True))
        fill = runner.load_config("final", intermediate=True)
        self.assertEqual(runner.load_config("final"), original)
        self.assertEqual(runner.load_config("final", n26=True), n26)
        self.assertNotIn(fill["subdir"], {original["subdir"], n26["subdir"]})

    def test_dry_run_does_not_launch(self):
        with patch.object(runner.subprocess, "run") as run:
            self.assertEqual(runner.main(["--intermediate", "--profile", "final", "--dry-run"]), 0)
        run.assert_not_called()

    def test_dispatches_only_intermediate_config_with_resume(self):
        with patch.object(runner.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(runner.main(["--intermediate", "--profile", "final"]), 0)
        command = run.call_args.args[0]
        self.assertIn(runner.CONFIG_INTERMEDIATE, command)
        self.assertNotIn(runner.CONFIG, command)
        self.assertNotIn(runner.CONFIG_N26, command)
        self.assertIn("--skip-existing", command)
        self.assertEqual(run.call_args.kwargs["env"][runner.PROFILE_ENV], "final")

    def test_incompatible_selections_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.load_config("final", intermediate=True, n26=True)
        with patch.object(runner.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                runner.main(["--intermediate", "--n26"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
