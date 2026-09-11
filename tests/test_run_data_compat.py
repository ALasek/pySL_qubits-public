import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.export import runDataLoad, runDataSave, run_store
from src.export.run_store import legacy_run_filename


class CompatPsi:
    I_S_Ef_fractionsT_runAv = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
    I_S_Ef_fractionsT_STD_runAv = np.zeros((2, 3))
    S_vn_runAv = np.array([0.0, 0.1, 0.2])
    T = 4
    dT = 0.1
    printT = 2
    nqubits = 2
    nqubits_E = 1
    nqubits_S = 1


def fake_data_dir(root):
    return lambda subdir="": os.path.join(root, subdir) if subdir else root


class RunDataCompatibilityTests(unittest.TestCase):
    def patch_data_dirs(self, root):
        return patch.multiple(
            run_store,
            get_data_dir=fake_data_dir(root),
        )

    def test_run_data_save_writes_v2_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                runDataSave.runDataSave(CompatPsi(), params, subdir="demo")
                out_dir = os.path.join(tmp, "demo")

                self.assertTrue(os.path.exists(os.path.join(out_dir, "run_index.jsonl")))
                self.assertFalse(os.path.exists(os.path.join(out_dir, legacy_run_filename(params))))

    def test_run_data_save_can_opt_into_legacy_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1, "save_legacy_rundata": True}
                runDataSave.runDataSave(CompatPsi(), params, subdir="demo")
                out_dir = os.path.join(tmp, "demo")

                self.assertTrue(os.path.exists(os.path.join(out_dir, legacy_run_filename(params))))

    def test_run_data_load_prefers_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)), patch.object(runDataLoad, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                runDataSave.runDataSave(CompatPsi(), params, subdir="demo")

                loaded = runDataLoad.runDataLoad({"theta": 0.1}, subdir="demo")

                self.assertEqual(len(loaded.matchdata), 1)
                self.assertEqual(loaded.matchdata[0][1].__class__.__name__, "StoredRunResult")

    def test_run_data_load_falls_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataLoad, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                params = {"theta": 0.1}
                with open(os.path.join(out_dir, legacy_run_filename(params)), "wb") as f:
                    pickle.dump([params, CompatPsi()], f)

                loaded = runDataLoad.runDataLoad({"theta": 0.1}, subdir="demo")

                self.assertEqual(len(loaded.matchdata), 1)
                self.assertIsInstance(loaded.matchdata[0][1], CompatPsi)

    def test_v2_result_supports_redundancy_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)), patch.object(runDataLoad, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                runDataSave.runDataSave(CompatPsi(), params, subdir="demo")
                loaded = runDataLoad.runDataLoad({"theta": 0.1}, subdir="demo")
                result = loaded.matchdata[0][1]

                self.assertTrue(hasattr(result, "redundancy_slope_at_T"))
                self.assertTrue(hasattr(result, "redundancy_fractionScore_at_T"))

    def test_v2_result_supports_plot_i_s(self):
        import matplotlib

        matplotlib.use("Agg", force=True)

        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)), patch.object(runDataLoad, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1, "psi_bias": 0}
                runDataSave.runDataSave(CompatPsi(), params, subdir="demo")
                loaded = runDataLoad.runDataLoad({"theta": 0.1}, subdir="demo")
                result = loaded.matchdata[0][1]

                with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
                    result.plot_I_S()

    def test_default_nonbatch_subdir_is_used_when_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.patch_data_dirs(tmp), patch.object(runDataSave, "get_data_dir", fake_data_dir(tmp)), patch.object(runDataLoad, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                runDataSave.runDataSave(CompatPsi(), params)

                out_dir = os.path.join(tmp, "nonbatch")
                self.assertTrue(os.path.exists(os.path.join(out_dir, "run_index.jsonl")))

                loaded = runDataLoad.runDataLoad({"theta": 0.1})
                self.assertEqual(len(loaded.matchdata), 1)


if __name__ == "__main__":
    unittest.main()
