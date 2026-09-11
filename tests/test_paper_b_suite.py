import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_paper_b_suite
from src.batch_utils import count_parameter_combinations, iter_parameter_combinations
from src.input.validation import PAPER_B_CASES, validate_params


class PaperBSuiteTests(unittest.TestCase):
    def test_profile_counts_and_staged_invariants(self):
        for profile, expected_count in (("smoke", 16), ("focused", 48), ("matched", 24)):
            config = run_paper_b_suite.load_config(profile)
            self.assertEqual(count_parameter_combinations(config["sweep"]), expected_count)
            params = config["params_default"]
            self.assertEqual(params["H_SE_Special"], "PaperB_staged")
            self.assertEqual(params["evolution_protocol"], "staged_write_store")
            self.assertEqual(params["staged_write_time"], 10.0)
            self.assertEqual(params["correlator_axis"], "X")
            self.assertEqual(params["correlator_sources"], [0])
            self.assertEqual(params["correlator_reference_time"], 10.0)
            self.assertTrue(params["compute_discord"])
            expected_normalization = "exact_rms" if profile == "matched" else "nominal"
            self.assertEqual(params["PaperB_field_normalization"], expected_normalization)
            if profile != "matched":
                self.assertEqual(config["sweep"]["PaperB_case"], list(PAPER_B_CASES))
            for point in iter_parameter_combinations(params, config["sweep"]):
                validate_params(point)

    def test_focused_profile_is_deliberately_bounded(self):
        config = run_paper_b_suite.load_config("focused")

        self.assertEqual(config["sweep"]["nqubits_E"], [8, 10])
        self.assertEqual(config["sweep"]["PaperB_field_strength"], [0.0, 0.2, 0.6])
        self.assertEqual(config["params_default"]["AverageOverRunsN"], 3)
        self.assertEqual(config["params_default"]["dT"], 0.01)
        self.assertEqual(config["params_default"]["T"], 60.0)
        self.assertEqual(config["params_default"]["fragment_sample_count"], 8)
        self.assertEqual(config["analysis"]["Tsample"], 60.0)
        self.assertEqual(config["analysis"]["setMinTime"], 20.0)

    def test_analysis_emits_primary_final_late_window_slice_for_each_size(self):
        self.assertEqual(len(run_paper_b_suite.analysis_commands("smoke")), 1)
        self.assertEqual(len(run_paper_b_suite.analysis_commands("focused")), 2)
        self.assertEqual(len(run_paper_b_suite.analysis_commands("matched")), 2)

        command = run_paper_b_suite.analysis_commands("focused")[0]
        self.assertIn("N8_final_late_window", command)
        self.assertIn("holevo_discord_fhalf", command)
        self.assertIn("holevo_redundancy_q", command)
        self.assertIn("discord_fhalf", run_paper_b_suite.load_config("focused")["analysis"]["metrics"][0])
        self.assertLess(command.index("holevo_redundancy_q"), command.index("qd_slope"))

    def test_dry_run_does_not_spawn_or_write_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_root = Path(temp_dir) / "paper_b_logs"
            with patch.object(run_paper_b_suite, "DEFAULT_LOG_ROOT", log_root), patch(
                "scripts.run_paper_b_suite.subprocess.Popen"
            ) as popen:
                result = run_paper_b_suite.main(["--dry-run", "--profile", "smoke", "--no-figures"])

            self.assertEqual(result, 0)
            popen.assert_not_called()
            self.assertFalse(log_root.exists())


if __name__ == "__main__":
    unittest.main()
