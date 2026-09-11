import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis_scripts.paper_a_submission_panels import _discover, build, parse_args


class PaperASubmissionAnalysisTests(unittest.TestCase):
    def test_discovery_excludes_incomplete_manifest_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "Paper_A_final_SubmissionMatchedLambda"
            run = batch / "runs" / "one"
            run.mkdir(parents=True)
            (run / "params.json").write_text("{}", encoding="utf-8")
            np.savez_compressed(run / "results.npz", placeholder=np.asarray([1]))
            (batch / "batch_manifest.json").write_text(
                json.dumps({"combo_count": 2, "status": "started"}),
                encoding="utf-8",
            )

            found, missing, completeness = _discover(root, "lambda")

            self.assertEqual(found, [])
            self.assertEqual(len(missing), 1)
            matched = next(row for row in completeness if row["batch"].endswith("SubmissionMatchedLambda"))
            self.assertEqual(matched["complete_points"], 1)
            self.assertTrue(matched["incomplete"])
            self.assertFalse(matched["included"])

    def test_builds_late_window_bundle_from_v2_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "Paper_A_final_LambdaCollapse" / "runs" / "synthetic"
            run_dir.mkdir(parents=True)
            params = {
                "AverageOverRunsN": 2,
                "H_SE_J": 0.1,
                "H_SE_Special": "Mironowicz_rand",
                "H_SE_bonds": "S_to_all",
                "H_SE_p": "Norm",
                "H_S_J": 0.0,
                "Mironowicz_alpha2": 0.0,
                "Mironowicz_h0": 0.0,
                "Mironowicz_theta": math.pi / 2,
                "compute_discord": True,
                "nqubits_E": 2,
                "nqubits_S": 1,
                "printT": 10,
                "psi_bias": 0.5,
                "psi_S_spec": "x+",
            }
            (run_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")
            shape = (3, 5, 2)
            holevo = np.full(shape, math.log(2.0))
            discord = np.full(shape, 0.01)
            quantiles = holevo[None, ...] * 0.95
            rho = np.tile(np.diag([0.5, 0.5]), (5, 1, 1)).astype(complex)
            fidelity = np.empty((2, 5, 2))
            fidelity[0] = 0.25
            fidelity[1] = 0.81
            np.savez_compressed(
                run_dir / "results.npz",
                Holevo_Z_S_Ef_fractionsT=holevo,
                Holevo_Z_S_Ef_fractionsT_quantiles=quantiles,
                Discord_Z_S_Ef_fractionsT=discord,
                SBS_fid_runs_1=fidelity,
                fragment_quantile_levels=np.asarray([0.1]),
                rhoS_T=rho,
            )
            output = root / "out"
            args = parse_args(
                [
                    "--data-root",
                    str(root),
                    "--output-dir",
                    str(output),
                    "--families",
                    "lambda",
                    "--bootstrap",
                    "20",
                ]
            )

            build(args)

            with (output / "paper_a_late_window_points.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["holevo_fhalf"]), 1.0)
            self.assertAlmostEqual(float(row["record_occupancy"]), 1.0)
            self.assertGreaterEqual(float(row["kappa_typ"]), float(row["kappa_ann"]))
            manifest = json.loads((output / "paper_a_analysis_manifest.json").read_text())
            self.assertEqual(manifest["loaded_point_count"], 1)
            self.assertIn("sqrt(F)", manifest["kappa_ann"])


if __name__ == "__main__":
    unittest.main()
