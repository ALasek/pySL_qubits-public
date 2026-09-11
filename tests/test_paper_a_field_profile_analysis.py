import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis_scripts.paper_a_field_profile_control import _load_point, _paired_rows


class PaperAFieldProfileAnalysisTests(unittest.TestCase):
    def _write_run(self, root: Path, profile: str, ratio: float, offset: float):
        run = root / f"{profile}-{ratio:g}"
        run.mkdir()
        params = {
            "AverageOverRunsN": 24,
            "H_EE_J": 0.0,
            "H_SE_J": 0.1,
            "H_SE_Special": "Mironowicz_rand",
            "H_SE_bonds": "S_to_all",
            "Mironowicz_ZZ_to_ZX_epsilon": 0.0,
            "Mironowicz_alpha2": 0.1 * ratio if profile == "random" else 0.0,
            "Mironowicz_epsilon": 0.0,
            "Mironowicz_epsilon2": 0.0,
            "Mironowicz_h0": 0.1 * ratio if profile == "uniform" else 0.0,
            "Mironowicz_theta": 0.0,
            "compute_discord": True,
            "field_geometry": "aligned",
            "field_profile": profile,
            "field_strength_ratio": ratio,
            "fragment_half_only": True,
            "fragment_sample_count": 64,
            "nqubits_E": 16,
            "nqubits_S": 1,
            "printT": 1,
            "psi_E_spec": "bias",
            "psi_S_spec": "x+",
            "psi_bias": 0.5,
            "run_seeds": list(range(24)),
            "store_fragment_samples": True,
        }
        (run / "params.json").write_text(json.dumps(params), encoding="utf-8")
        shape = (17, 61, 24)
        holevo = np.full(shape, 0.95 * math.log(2.0) + offset)
        discord = np.full(shape, 0.02 * math.log(2.0))
        quantiles = np.stack((holevo * 0.95, holevo, holevo * 1.02))
        rho = np.tile(np.diag([0.5, 0.5]), (61, 1, 1)).astype(complex)
        np.savez_compressed(
            run / "results.npz",
            Holevo_Z_S_Ef_fractionsT=holevo,
            Holevo_Z_S_Ef_fractionsT_quantiles=quantiles,
            Discord_Z_S_Ef_fractionsT=discord,
            SBS_fid_runs_1=np.full((16, 61, 24), 0.25),
            fragment_quantile_levels=np.asarray([0.1, 0.5, 0.9]),
            norms=np.ones(61),
            rhoS_T=rho,
        )
        point, health = _load_point(run, (30.0, 60.0), np.random.default_rng(1), 20, 3)
        return point, health

    def test_load_and_paired_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uniform, health = self._write_run(root, "uniform", 0.5, 0.0)
            random, _ = self._write_run(root, "random", 0.5, 0.01)
            self.assertAlmostEqual(health["max_norm_error"], 0.0)
            self.assertAlmostEqual(uniform.metrics["holevo_fhalf"][0], 0.95)
            rows = _paired_rows([uniform, random], np.random.default_rng(2), 20)
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]["delta_holevo_fhalf"], 0.0)


if __name__ == "__main__":
    unittest.main()
