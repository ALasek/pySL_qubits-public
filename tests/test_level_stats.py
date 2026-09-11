import math
import unittest

import numpy as np

from eigensolver.level_stats import adjacent_gap_ratio, level_stats_for_params, sector_info
from batch_configs.defaults import build_batch_params_default


def small_params(theta=np.pi / 2, nqubits_E=3):
    params = build_batch_params_default()
    params.update(
        {
            "H_SE_Special": "Mironowicz_rand_EE",
            "Mironowicz_theta": theta,
            "Mironowicz_alpha2": 0.2,
            "H_EE_J": 0.1,
            "H_EE_p": "Const",
            "nqubits_E": nqubits_E,
            "seed": 7,
        }
    )
    return params


class LevelStatsTests(unittest.TestCase):
    def test_adjacent_gap_ratio(self):
        stats = adjacent_gap_ratio([0, 1, 3, 6], trim_fraction=0)

        self.assertTrue(np.allclose(stats["ratios"], [0.5, 2 / 3]))
        self.assertAlmostEqual(stats["mean_r"], (0.5 + 2 / 3) / 2)
        self.assertEqual(stats["n_ratios"], 2)

    def test_sector_info_uses_middle_sector(self):
        info = sector_info(4, "middle")

        self.assertEqual(info["sector_k"], 2)
        self.assertEqual(info["sector_sz"], 0.0)
        self.assertEqual(info["sector_dim"], 6)

    def test_level_stats_uses_sector_when_sz_is_conserved(self):
        stats = level_stats_for_params(small_params(theta=np.pi / 2), trim_fraction=0)

        self.assertTrue(stats["conserved"])
        self.assertEqual(stats["sector_k"], 2)
        self.assertEqual(stats["sector_dim"], 6)
        self.assertEqual(stats["n_levels"], 6)
        self.assertTrue(stats["is_real"])
        self.assertEqual(stats["eig_dtype"], "float32")
        self.assertFalse(math.isnan(stats["mean_r"]))

    def test_level_stats_uses_full_spectrum_when_not_conserved(self):
        stats = level_stats_for_params(
            small_params(theta=0, nqubits_E=2),
            trim_fraction=0,
            max_full_dim=16,
        )

        self.assertFalse(stats["conserved"])
        self.assertIsNone(stats["sector_k"])
        self.assertEqual(stats["full_dim"], 8)
        self.assertEqual(stats["n_levels"], 8)
        self.assertEqual(stats["eig_dtype"], "float32")

    def test_auto_dtype_uses_complex64_for_complex_hamiltonian(self):
        params = small_params(theta=np.pi / 2, nqubits_E=2)
        params["Mironowicz_H_E"] = "Y"

        stats = level_stats_for_params(params, trim_fraction=0)

        self.assertFalse(stats["is_real"])
        self.assertEqual(stats["eig_dtype"], "complex64")

    def test_requested_sector_requires_conservation(self):
        with self.assertRaisesRegex(ValueError, "Sz is not conserved"):
            level_stats_for_params(
                small_params(theta=0, nqubits_E=2),
                sector="middle",
                max_full_dim=16,
            )


if __name__ == "__main__":
    unittest.main()
