import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_paper_a_suite
from src.batch_utils import count_parameter_combinations, iter_parameter_combinations
from src.input.validation import validate_params


class PaperASuiteTests(unittest.TestCase):
    def test_profile_counts_and_common_invariants(self):
        expected_counts = {
            "smoke": {"lambda_collapse": 25, "uniform_field": 18, "disorder_field": 18, "stability_map": 27},
            "final": {"lambda_collapse": 75, "uniform_field": 102, "disorder_field": 102, "stability_map": 363},
        }
        for profile, counts in expected_counts.items():
            for job, expected_count in counts.items():
                config = run_paper_a_suite.load_job_config(job, profile)
                self.assertEqual(count_parameter_combinations(config["sweep"]), expected_count)
                params = config["params_default"]
                self.assertEqual(params["seed"], 1337)
                self.assertEqual(params["H_SE_Special"], "Mironowicz_rand")
                self.assertEqual(params["H_SE_J"], 0.1)
                self.assertEqual(params["H_EE_J"], 0.0)
                self.assertEqual(params["QREmaxFragSize"], 1)
                self.assertTrue(params["fragment_reuse_samples"])
                self.assertTrue(params["fragment_half_only"])
                self.assertEqual(params["fragment_quantiles"], [0.1, 0.5])
                self.assertTrue(params["compute_discord"])
                self.assertFalse(params["store_psi"])
                self.assertFalse(params["store_correlators"])
                self.assertEqual(params["plot_mode"], "none")
                self.assertEqual(config["analysis"]["holevo_delta"], 0.1)
                self.assertEqual(config["analysis"]["holevo_quantile"], 0.1)
                for params_at_point in iter_parameter_combinations(params, config["sweep"]):
                    validate_params(params_at_point)

    def test_profile_specific_grids_and_durations(self):
        lambda_final = run_paper_a_suite.load_job_config("lambda_collapse", "final")
        uniform_final = run_paper_a_suite.load_job_config("uniform_field", "final")
        disorder_smoke = run_paper_a_suite.load_job_config("disorder_field", "smoke")

        self.assertEqual(lambda_final["sweep"]["nqubits_E"], [8, 12, 16])
        self.assertEqual(lambda_final["params_default"]["T"], 40)
        self.assertEqual(uniform_final["params_default"]["T"], 60)
        self.assertAlmostEqual(uniform_final["sweep"]["Mironowicz_h0"][0], -0.4)
        self.assertAlmostEqual(uniform_final["sweep"]["Mironowicz_h0"][-1], 0.4)
        np.testing.assert_allclose(disorder_smoke["sweep"]["Mironowicz_alpha2"], [0.0, 0.1, 0.3])
        self.assertTrue(lambda_final["subdir"].startswith("Paper_A_final_"))

    def test_lambda_grid_contains_repeated_fixed_lambda_pairs(self):
        config = run_paper_a_suite.load_job_config("lambda_collapse", "smoke")
        lambdas = []
        for bias in config["sweep"]["psi_bias"]:
            beta = 2 * math.acos(math.sqrt(bias))
            for theta in config["sweep"]["Mironowicz_theta"]:
                lambdas.append(round(math.cos(beta + theta) ** 2, 12))

        for target in (0.0, 0.5, 1.0):
            self.assertGreaterEqual(lambdas.count(target), 2)

    def test_dry_run_does_not_spawn_or_write_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_root = Path(temp_dir) / "paper_a_logs"
            with patch.object(run_paper_a_suite, "DEFAULT_LOG_ROOT", log_root), patch(
                "scripts.run_paper_a_suite.subprocess.Popen"
            ) as popen:
                result = run_paper_a_suite.main(
                    ["--dry-run", "--profile", "smoke", "--jobs", "uniform_field", "--no-figures"]
                )

            self.assertEqual(result, 0)
            popen.assert_not_called()
            self.assertFalse(log_root.exists())

    def test_manifest_only_command_uses_pyslbatch_manifest_flag(self):
        args = type(
            "Args",
            (),
            {"workers": 2, "gpu_ids": "0,1", "manifest_only": True, "skip_existing": True, "worker_stagger_seconds": 0},
        )()
        command = run_paper_a_suite.batch_command("uniform_field", args)
        self.assertIn("--manifest-only", command)
        self.assertNotIn("--skip-existing", command)

    def test_analysis_commands_slice_all_extra_sweep_dimensions(self):
        self.assertEqual(len(run_paper_a_suite.analysis_commands("lambda_collapse", "final")), 3)
        self.assertEqual(len(run_paper_a_suite.analysis_commands("uniform_field", "final")), 2)
        self.assertEqual(len(run_paper_a_suite.analysis_commands("disorder_field", "final")), 2)
        self.assertEqual(len(run_paper_a_suite.analysis_commands("stability_map", "final")), 3)


if __name__ == "__main__":
    unittest.main()
