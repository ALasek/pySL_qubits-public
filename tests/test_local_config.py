import json
import os
import tempfile
import unittest

from src.analysis.local_config import apply_config_defaults, load_local_config_with_keys


class LocalConfigTests(unittest.TestCase):
    def test_load_local_config_with_keys_tracks_only_json_keys(self):
        defaults = {"Tsample": 500, "plotData": False}

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "analysis.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"Tsample": 60}, f)

            config, keys = load_local_config_with_keys(defaults, path, "test")

        self.assertEqual(config, {"Tsample": 60, "plotData": False})
        self.assertEqual(keys, {"Tsample"})

    def test_load_local_config_with_keys_ignores_null_overrides(self):
        defaults = {"Tsample": 500, "plotData": False}

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "analysis.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"Tsample": None, "plotData": True}, f)

            config, keys = load_local_config_with_keys(defaults, path, "test")

        self.assertEqual(config, {"Tsample": None, "plotData": True})
        self.assertEqual(keys, {"plotData"})

    def test_apply_config_defaults_preserves_protected_keys(self):
        config = {"x": "local_x", "y": "default_y", "Tsample": 500}
        defaults = {"x": "batch_x", "y": "batch_y", "Tsample": 60}

        merged = apply_config_defaults(
            config,
            defaults,
            protected_keys={"x"},
            keys=("x", "y", "Tsample"),
        )

        self.assertEqual(merged, {"x": "local_x", "y": "batch_y", "Tsample": 60})


if __name__ == "__main__":
    unittest.main()
