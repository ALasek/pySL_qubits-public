import os
import shutil
import unittest

from src.batch_utils import (
    build_batch_manifest,
    get_manifest_path,
    load_batch_manifest,
    normalize_sweep_values,
    write_batch_manifest,
)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class BatchManifestTests(unittest.TestCase):
    def test_normalizes_scalar_like_sweep_values(self):
        sweep = {
            "theta": [FakeScalar(0.0), FakeScalar(0.02), FakeScalar(0.04)],
            "alpha": [FakeScalar(0.1)],
            "label": ["Z"],
        }

        normalized = normalize_sweep_values(sweep)

        self.assertEqual(normalized["theta"], [0.0, 0.02, 0.04])
        self.assertEqual(normalized["alpha"], [0.1])
        self.assertEqual(normalized["label"], ["Z"])

    def test_build_manifest_contains_sweep_and_analysis(self):
        manifest = build_batch_manifest(
            params_default={"nqubits_E": FakeScalar(15)},
            loops_dist={"x": [1, 2], "y": [3]},
            subdir="demo",
            analysis={"x": "x", "y": "y"},
        )

        self.assertEqual(manifest["subdir"], "demo")
        self.assertEqual(manifest["combo_count"], 2)
        self.assertEqual(manifest["params_default"]["nqubits_E"], 15)
        self.assertEqual(manifest["analysis"]["x"], "x")
        self.assertEqual(manifest["schema_version"], 2)
        self.assertIn("source_provenance", manifest)

    def test_build_manifest_contains_explicit_cases(self):
        manifest = build_batch_manifest(
            params_default={"fixed": True},
            cases=[{"p": FakeScalar(0.1), "theta": 0.2}, {"p": 0.3, "theta": 0.4}],
            subdir="matched",
        )

        self.assertEqual(manifest["combo_count"], 2)
        self.assertEqual(manifest["case_keys"], ["p", "theta"])
        self.assertEqual(manifest["cases"][0], {"p": 0.1, "theta": 0.2})
        self.assertNotIn("sweep_keys", manifest)

    def test_build_manifest_rejects_ambiguous_parameter_mode(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_batch_manifest(params_default={}, subdir="missing")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_batch_manifest(
                params_default={},
                loops_dist={"x": [1]},
                cases=[{"x": 1}],
                subdir="both",
            )

    def test_write_and_load_manifest(self):
        subdir = "__unit_manifest__"
        path = get_manifest_path(subdir)
        root = os.path.dirname(path)
        shutil.rmtree(root, ignore_errors=True)
        try:
            write_batch_manifest(
                params_default={"base": True},
                loops_dist={"x": [1, 2]},
                subdir=subdir,
                analysis={"x": "x"},
                status="unit",
            )

            loaded = load_batch_manifest(subdir)
            self.assertEqual(loaded["status"], "unit")
            self.assertEqual(loaded["sweep"]["x"], [1, 2])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
