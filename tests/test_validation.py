import unittest

from src.input.validation import derive_time_grid, validate_params


def make_params():
    return {
        "dtypepbits": 32,
        "AverageOverRunsN": 1,
        "nqubits_S": 1,
        "nqubits_E": 2,
        "QREmaxFragSize": 2,
        "psi_E_spec": "bias",
        "psi_bias": 0.5,
        "dT": 0.25,
        "T": 1.0,
        "printT": 0.5,
        "p_noise_Gate": 0,
        "noiseT_Gate": 0,
        "noiseT": 0,
        "seed": "TIME",
    }


class ValidationTests(unittest.TestCase):
    def test_valid_params_and_time_grid(self):
        params = make_params()
        validate_params(params)
        grid = derive_time_grid(params)
        self.assertEqual(grid["total_steps"], 4)
        self.assertEqual(grid["print_interval_steps"], 2)
        self.assertEqual(grid["print_sample_count"], 3)
        self.assertIsNone(grid["noise_interval_steps"])

    def test_rejects_non_integral_print_grid(self):
        params = make_params()
        params["printT"] = 0.3
        with self.assertRaisesRegex(ValueError, "printT must be an integer multiple"):
            validate_params(params)

    def test_rejects_invalid_noise_gate_schedule(self):
        params = make_params()
        params["p_noise_Gate"] = 0.25
        with self.assertRaisesRegex(ValueError, "noiseT_Gate must be a positive integer"):
            validate_params(params)

    def test_rejects_unsupported_dtype(self):
        params = make_params()
        params["dtypepbits"] = 16
        with self.assertRaisesRegex(ValueError, "dtypepbits must be 32 or 64"):
            validate_params(params)

    def test_rejects_fragment_size_larger_than_environment(self):
        params = make_params()
        params["QREmaxFragSize"] = 3
        with self.assertRaisesRegex(ValueError, "QREmaxFragSize must be less than or equal to nqubits_E"):
            validate_params(params)

    def test_rejects_invalid_save_figures_mode(self):
        params = make_params()
        params["save_figures"] = "always"
        with self.assertRaisesRegex(ValueError, "save_figures must be one of"):
            validate_params(params)

    def test_rejects_unsupported_multiple_system_qubits(self):
        params = make_params()
        params["nqubits_S"] = 2
        with self.assertRaisesRegex(ValueError, "nqubits_S must be 1"):
            validate_params(params)

    def test_accepts_none_plot_mode(self):
        params = make_params()
        params["plot_mode"] = "none"
        validate_params(params)

    def test_accepts_fixed_seed_with_average_over_runs(self):
        params = make_params()
        params["seed"] = 1337
        params["AverageOverRunsN"] = 3
        validate_params(params)

    def test_accepts_store_psi_with_average_over_runs(self):
        params = make_params()
        params["AverageOverRunsN"] = 2
        params["store_psi"] = True
        validate_params(params)

    def test_rejects_invalid_store_psi_flags(self):
        params = make_params()
        params["store_psi"] = "yes"
        with self.assertRaisesRegex(ValueError, "store_psi"):
            validate_params(params)

        params = make_params()
        params["store_psi_on_disk"] = "yes"
        with self.assertRaisesRegex(ValueError, "store_psi_on_disk"):
            validate_params(params)

    def test_rejects_invalid_seed(self):
        params = make_params()
        params["seed"] = "not-a-seed"
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            validate_params(params)

    def test_rejects_invalid_profile_evolution_limit(self):
        params = make_params()
        params["profile_evolution_max_steps"] = -1
        with self.assertRaisesRegex(ValueError, "profile_evolution_max_steps"):
            validate_params(params)

    def test_accepts_grouped_h_precompute_mode(self):
        params = make_params()
        params["H_precompute_mode"] = "grouped_cpu"
        params["H_precompute_group_size"] = 4
        validate_params(params)

    def test_rejects_invalid_h_precompute_mode(self):
        params = make_params()
        params["H_precompute_mode"] = "gpu_magic"
        with self.assertRaisesRegex(ValueError, "H_precompute_mode"):
            validate_params(params)

    def test_rejects_invalid_h_precompute_group_size(self):
        params = make_params()
        params["H_precompute_group_size"] = 0
        with self.assertRaisesRegex(ValueError, "H_precompute_group_size"):
            validate_params(params)

    def test_rejects_invalid_gpu_memory_logging_flag(self):
        params = make_params()
        params["log_gpu_memory"] = "yes"
        with self.assertRaisesRegex(ValueError, "log_gpu_memory"):
            validate_params(params)

    def test_rejects_invalid_legacy_rundata_flag(self):
        params = make_params()
        params["save_legacy_rundata"] = "yes"
        with self.assertRaisesRegex(ValueError, "save_legacy_rundata"):
            validate_params(params)

    def test_rejects_invalid_compute_discord_flag(self):
        params = make_params()
        params["compute_discord"] = "no"
        with self.assertRaisesRegex(ValueError, "compute_discord"):
            validate_params(params)

    def test_validates_fragment_sampling_controls(self):
        params = make_params()
        params.update(
            {
                "fragment_sample_count": 12,
                "fragment_reuse_samples": True,
                "fragment_half_only": True,
                "fragment_quantiles": [0.1, 0.5],
                "Mironowicz_h0": -0.25,
            }
        )
        validate_params(params)

    def test_rejects_invalid_fragment_sampling_controls(self):
        params = make_params()
        params["fragment_sample_count"] = 0
        with self.assertRaisesRegex(ValueError, "fragment_sample_count"):
            validate_params(params)

        params = make_params()
        params["fragment_quantiles"] = [1.1]
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_params(params)

        params = make_params()
        params["Mironowicz_h0"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Mironowicz_h0"):
            validate_params(params)

    def test_validates_stored_fragment_samples(self):
        params = make_params()
        params.update(
            {
                "fragment_sample_count": 4,
                "fragment_reuse_samples": True,
                "store_fragment_samples": True,
            }
        )
        validate_params(params)

        params["fragment_reuse_samples"] = False
        with self.assertRaisesRegex(ValueError, "fragment_reuse_samples"):
            validate_params(params)

        params = make_params()
        params["store_fragment_samples"] = True
        with self.assertRaisesRegex(ValueError, "finite fragment_sample_count"):
            validate_params(params)

    def test_accepts_staged_paper_b_protocol(self):
        params = make_params()
        params.update(
            {
                "T": 2.0,
                "H_SE_Special": "PaperB_staged",
                "PaperB_case": "zx_gaussian_chain",
                "PaperB_field_strength": 0.6,
                "evolution_protocol": "staged_write_store",
                "staged_write_time": 1.0,
                "correlator_axis": "X",
                "correlator_sources": [0],
                "correlator_reference_time": 1.0,
            }
        )

        validate_params(params)

    def test_rejects_staged_write_time_off_print_grid(self):
        params = make_params()
        params.update(
            {
                "T": 2.0,
                "evolution_protocol": "staged_write_store",
                "staged_write_time": 0.75,
            }
        )

        with self.assertRaisesRegex(ValueError, "staged_write_time must be an integer multiple of dT=0.5"):
            validate_params(params)

    def test_rejects_invalid_correlator_source(self):
        params = make_params()
        params["correlator_sources"] = [2]

        with self.assertRaisesRegex(ValueError, "less than nqubits_E"):
            validate_params(params)


if __name__ == "__main__":
    unittest.main()
