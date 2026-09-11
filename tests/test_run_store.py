import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.export import run_store
from src.export.run_store import (
    canonicalize_value,
    compute_run_id,
    load_runs_matching,
    save_run,
)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakePsi:
    def __init__(self):
        self.params = {"theta": 0.1}
        self.I_S_Ef_fractionsT_runAv = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
        self.I_S_Ef_fractionsT_STD_runAv = np.zeros((2, 3))
        self.S_vn_runAv = np.array([0.0, 0.2, 0.4])
        self.printT = 2
        self.T = 4
        self.dT = 0.1
        self.nqubits = 2
        self.nqubits_E = 1
        self.nqubits_S = 1
        self.norms = [1.0, 0.9999, 1.0001]
        self.modified_norms = [1.0, 1.0, 1.0]
        self.fragment_quantile_levels = np.array([0.1])
        self.fragment_sample_members = np.array([[[[-1]], [[-1]]], [[[1]], [[1]]]])
        self.Holevo_Z_S_Ef_fractionsT = np.ones((2, 3, 1))
        self.Holevo_Z_S_Ef_fractionsT_quantiles = np.ones((1, 2, 3, 1)) * 0.8
        self.Holevo_Z_S_Ef_fractionsT_quantiles_runAv = np.ones((1, 2, 3)) * 0.8
        self.Holevo_Z_S_Ef_fractionsT_samples = np.ones((2, 3, 1, 1)) * 0.75
        self.SBS_qre = [None, np.ones((1, 3))]
        self.SBS_trace_dist = [None, np.ones((1, 3)) * 0.7]
        self.SBS_fid = [None, np.ones((1, 3)) * 0.25]
        self.SBS_fid_runs = [None, np.ones((1, 3, 1)) * 0.25]
        self.branch_energy_density_gap_runAv = 1e-6
        self.branch_energy_density_std_runAv = np.array([0.2, 0.3])
        self.corr_total_commutator_weight_T_runAv = np.array([0.0, 0.1, 0.2])


def fake_data_dir(root):
    return lambda subdir="": os.path.join(root, subdir) if subdir else root


class RunStoreTests(unittest.TestCase):
    def test_canonical_param_normalization(self):
        params = {
            "b": FakeScalar(2),
            "a": [FakeScalar(0.1), {"z": FakeScalar(3)}],
            "label": "Z",
        }

        self.assertEqual(
            canonicalize_value(params),
            {"a": [0.1, {"z": 3}], "b": 2, "label": "Z"},
        )

    def test_run_id_is_stable_for_canonical_params(self):
        left = {"theta": 0.1, "nested": {"a": 1, "b": 2}}
        right = {"nested": {"b": 2, "a": 1}, "theta": 0.1}

        self.assertEqual(compute_run_id(left), compute_run_id(right))
        self.assertNotEqual(compute_run_id(left), compute_run_id({**left, "theta": 0.2}))

    def test_save_and_load_v2_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1, "label": "Z"}
                save_result = save_run(FakePsi(), params, subdir="demo", write_legacy=False)
                run_id = save_result["run_id"]
                run_dir = os.path.join(tmp, "demo", "runs", run_id)

                self.assertTrue(os.path.exists(os.path.join(run_dir, "params.json")))
                self.assertTrue(os.path.exists(os.path.join(run_dir, "metadata.json")))
                self.assertTrue(os.path.exists(os.path.join(run_dir, "results.npz")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "demo", "run_index.jsonl")))

                matches = load_runs_matching({"theta": 0.1}, subdir="demo", allow_legacy=False)
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0][0]["label"], "Z")
                self.assertEqual(matches[0][1].printT, 2)
                np.testing.assert_array_equal(matches[0][1].norms, [1.0, 0.9999, 1.0001])
                np.testing.assert_array_equal(matches[0][1].modified_norms, [1.0, 1.0, 1.0])
                np.testing.assert_array_equal(matches[0][1].SBS_fid_1, np.ones((1, 3)) * 0.25)
                self.assertEqual(matches[0][1].SBS_fid_runs_1.shape, (1, 3, 1))
                self.assertEqual(matches[0][1].Holevo_Z_S_Ef_fractionsT.shape, (2, 3, 1))
                self.assertEqual(matches[0][1].Holevo_Z_S_Ef_fractionsT_quantiles.shape, (1, 2, 3, 1))
                self.assertEqual(matches[0][1].fragment_sample_members.shape, (2, 2, 1, 1))
                self.assertEqual(matches[0][1].Holevo_Z_S_Ef_fractionsT_samples.shape, (2, 3, 1, 1))
                self.assertEqual(matches[0][1].branch_energy_density_gap_runAv, 1e-6)
                np.testing.assert_array_equal(
                    matches[0][1].corr_total_commutator_weight_T_runAv,
                    [0.0, 0.1, 0.2],
                )

    def test_v2_loader_matches_floats_with_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                save_run(FakePsi(), {"theta": 0.1 + 1e-13, "label": "Z"}, subdir="demo", write_legacy=False)

                matches = load_runs_matching({"theta": 0.1, "label": "Z"}, subdir="demo", allow_legacy=False)

                self.assertEqual(len(matches), 1)

    def test_missing_optional_fields_do_not_break_load(self):
        class MinimalPsi:
            I_S_Ef_fractionsT_runAv = np.array([[0.1, 0.2], [0.3, 0.4]])
            printT = 1

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                save_run(MinimalPsi(), {"theta": 0.1}, subdir="demo", write_legacy=False)

                matches = load_runs_matching({"theta": 0.1}, subdir="demo", allow_legacy=False)

                self.assertEqual(len(matches), 1)
                self.assertFalse(hasattr(matches[0][1], "rhoS_T"))

    def test_overwrite_preserves_run_scoped_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                save_result = save_run(FakePsi(), params, subdir="demo", write_legacy=False)
                run_dir = os.path.join(tmp, "demo", "runs", save_result["run_id"])
                figure_dir = os.path.join(run_dir, "figs")
                os.makedirs(figure_dir)
                figure_path = os.path.join(figure_dir, "result.png")
                with open(figure_path, "wb") as f:
                    f.write(b"figure")

                save_run(FakePsi(), params, subdir="demo", write_legacy=False)

                self.assertTrue(os.path.exists(figure_path))

    def test_legacy_pickle_memory_error_does_not_break_v2_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                params = {"theta": 0.1}
                with patch.object(run_store, "save_legacy_rundata", side_effect=MemoryError("too large")):
                    save_result = save_run(FakePsi(), params, subdir="demo", write_legacy=True)

                run_dir = os.path.join(tmp, "demo", "runs", save_result["run_id"])
                self.assertTrue(os.path.exists(os.path.join(run_dir, "results.npz")))
                self.assertIsNone(save_result["legacy_path"])
                self.assertIn("MemoryError", save_result["legacy_error"])

    def test_save_and_load_per_realization_psi_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                source_dir = tempfile.mkdtemp(dir=tmp)
                records = []
                arrays = []
                for run_index in range(2):
                    arr = np.zeros((3, 2, 4), dtype=np.float32)
                    arr[:, 0, :] = run_index + 1
                    arr[:, 1, :] = 10 * (run_index + 1)
                    path = os.path.join(source_dir, f"src_run_{run_index}.npy")
                    np.save(path, arr)
                    arrays.append(arr)
                    records.append(
                        {
                            "run_index": run_index,
                            "seed": 100 + run_index,
                            "path": path,
                            "shape": list(arr.shape),
                            "dtype": "float32",
                            "format": "npy",
                            "representation": "physical_psi_real_imag",
                            "written_indices": [0, 1, 2],
                            "sample_times": [0.0, 2.0, 4.0],
                        }
                    )

                psi = FakePsi()
                psi.psi_snapshot_records = records
                save_result = save_run(psi, {"theta": 0.1}, subdir="demo", write_legacy=False)
                run_id = save_result["run_id"]
                run_dir = os.path.join(tmp, "demo", "runs", run_id)
                metadata_path = os.path.join(run_dir, "psi_snapshots.json")

                self.assertTrue(os.path.exists(metadata_path))
                with open(metadata_path, encoding="utf-8") as f:
                    snapshot_metadata = json.load(f)
                self.assertEqual(snapshot_metadata["count"], 2)
                self.assertFalse(os.path.exists(records[0]["path"]))
                self.assertTrue(os.path.exists(os.path.join(run_dir, "psi_snapshots", "run_00000_psi_snapshots.npy")))
                self.assertIn("psi_snapshots", save_result["row"]["paths"])

                matches = load_runs_matching({"theta": 0.1}, subdir="demo", allow_legacy=False)
                loaded = matches[0][1]
                psi_t = loaded.get_stored_psi(1, run_index=1)

                expected = arrays[1][1, 0].astype(np.complex64) + 1j * arrays[1][1, 1].astype(np.complex64)
                np.testing.assert_allclose(psi_t, expected)


if __name__ == "__main__":
    unittest.main()

