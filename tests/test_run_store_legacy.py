import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.export import run_store
from src.export.run_store import legacy_run_filename, migrate_legacy_subdir


class LegacyPsi:
    I_S_Ef_fractionsT_runAv = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
    I_S_Ef_fractionsT_STD_runAv = np.zeros((2, 3))
    S_vn_runAv = np.array([0.0, 0.1, 0.2])
    T = 4
    dT = 0.1
    printT = 2


def fake_data_dir(root):
    return lambda subdir="": os.path.join(root, subdir) if subdir else root


def write_pickle(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


class LegacyMigrationTests(unittest.TestCase):
    def test_current_legacy_payload_migrates_to_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                params = {"theta": 0.1}
                write_pickle(os.path.join(out_dir, legacy_run_filename(params)), [params, LegacyPsi()])

                report = migrate_legacy_subdir("demo")

                self.assertEqual(len(report["migrated"]), 1)
                self.assertTrue(os.path.exists(os.path.join(out_dir, "run_index.jsonl")))

    def test_old_tuple_payload_migrates_to_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                params = {"theta": 0.2}
                payload = [
                    params,
                    np.array([[0.1, 0.2], [0.2, 0.4]]),
                    np.zeros((2, 2)),
                    np.array([0.0, 0.1]),
                    2,
                    0.1,
                    1,
                ]
                write_pickle(os.path.join(out_dir, legacy_run_filename(params)), payload)

                report = migrate_legacy_subdir("demo")

                self.assertEqual(len(report["migrated"]), 1)
                self.assertEqual(report["migrated"][0]["layout"], "old_tuple")

    def test_invalid_legacy_file_is_recorded_as_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                with open(os.path.join(out_dir, "bad.rundata"), "wb") as f:
                    f.write(b"not a pickle")

                report = migrate_legacy_subdir("demo")

                self.assertEqual(len(report["skipped"]), 1)
                self.assertEqual(report["skipped"][0]["reason"], "error")

    def test_migration_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                params = {"theta": 0.1}
                legacy_path = os.path.join(out_dir, legacy_run_filename(params))
                write_pickle(legacy_path, [params, LegacyPsi()])

                migrate_legacy_subdir("demo")

                self.assertTrue(os.path.exists(legacy_path))

    def test_dry_run_does_not_write_v2_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)):
                out_dir = os.path.join(tmp, "demo")
                os.makedirs(out_dir)
                params = {"theta": 0.1}
                write_pickle(os.path.join(out_dir, legacy_run_filename(params)), [params, LegacyPsi()])

                report = migrate_legacy_subdir("demo", dry_run=True)

                self.assertEqual(len(report["migrated"]), 1)
                self.assertFalse(os.path.exists(os.path.join(out_dir, "runs")))
                self.assertFalse(os.path.exists(os.path.join(out_dir, "migration_report.json")))


if __name__ == "__main__":
    unittest.main()

