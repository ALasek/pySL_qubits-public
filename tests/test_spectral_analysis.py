import unittest

import numpy as np

from batch_configs.defaults import build_batch_params_default
from src.analysis.spectral_analysis import analyze_spectral_batch


def tiny_manifest():
    params = build_batch_params_default()
    params.update(
        {
            "H_SE_Special": "Mironowicz_rand_EE",
            "Mironowicz_theta": np.pi / 2,
            "H_EE_p": "Const",
            "nqubits_E": 2,
            "seed": 11,
        }
    )
    return {
        "subdir": "__unit_spectral__",
        "params_default": params,
        "sweep": {
            "H_EE_J": [0.0, 0.2],
            "Mironowicz_alpha2": [0.0, 0.5],
        },
        "analysis": {
            "x": "H_EE_J",
            "y": "Mironowicz_alpha2",
        },
    }


class SpectralAnalysisTests(unittest.TestCase):
    def test_analyze_spectral_batch_builds_2d_grid(self):
        result = analyze_spectral_batch(tiny_manifest(), trim_fraction=0, max_full_dim=16)

        self.assertEqual(result["x_axis"], "H_EE_J")
        self.assertEqual(result["y_axis"], "Mironowicz_alpha2")
        self.assertEqual(len(result["results"]["mean_r"]), 2)
        self.assertEqual(len(result["results"]["mean_r"][0]), 2)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["results"]["conserved"][0][0])
        self.assertTrue(result["results"]["is_real"][0][0])
        self.assertEqual(result["results"]["eig_dtype"][0][0], "float32")
        self.assertEqual(result["results"]["backend"][0][0], "cpu")

    def test_analyze_spectral_batch_records_full_dim_errors(self):
        manifest = tiny_manifest()
        manifest["params_default"]["Mironowicz_theta"] = 0

        result = analyze_spectral_batch(manifest, trim_fraction=0, max_full_dim=4)

        self.assertEqual(len(result["errors"]), 4)
        self.assertTrue(result["results"]["bad"][0][0])


if __name__ == "__main__":
    unittest.main()
