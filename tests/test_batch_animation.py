import math
import unittest

import numpy as np

from src.analysis.batch_animation import build_time_cube


class FakeTimePsi:
    printT = 2.0

    def __init__(self, slopes):
        self.slopes = slopes
        self.calls = []
        self.min_times = []
        self.I_S_Ef_fractionsT_runAv = np.zeros((3, len(slopes)))
        self.nqubitsE = 2
        self.Discord_Z_S_Ef_fractionsT_runAv = np.array(
            [
                [9.0 for _ in slopes],
                [value + 10.0 if value is not None else np.nan for value in slopes],
                [8.0 for _ in slopes],
            ],
            dtype=float,
        )

    def redundancy_slope_at_T(self, T, cutoff=0.5, minTime=0):
        if T < 0:
            raise AssertionError("animation must not use best-time slope search")
        self.calls.append(T)
        self.min_times.append(minTime)
        value = self.slopes[T]
        if value is None:
            return None
        return [value, T, 1.0, 1.0, [0]]


class BatchAnimationTests(unittest.TestCase):
    def test_time_cube_uses_fixed_time_slope_frames(self):
        manifest = {
            "subdir": "demo",
            "params_default": {"fixed": 1},
            "sweep": {"theta": [0.0, 1.0], "alpha": [0.0], "fixed": [1]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        psis = {
            0.0: FakeTimePsi([0.1, 0.2, 0.3]),
            1.0: FakeTimePsi([1.1, 1.2, 1.3]),
        }

        def loader(filter_dict, *_args, **_kwargs):
            psi = psis[filter_dict["theta"]]
            return [[filter_dict, psi]]

        result = build_time_cube(manifest, loader=loader, set_min_time=0)

        self.assertEqual(result["cube"].shape, (3, 2, 1))
        np.testing.assert_allclose(result["times"], [0.0, 2.0, 4.0])
        np.testing.assert_allclose(result["cube"][:, 0, 0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(result["cube"][:, 1, 0], [1.1, 1.2, 1.3])
        self.assertEqual(psis[0.0].calls, [0, 1, 2])
        self.assertEqual(psis[1.0].calls, [0, 1, 2])

    def test_default_set_min_time_ignores_static_best_time_manifest_cutoff(self):
        manifest = {
            "subdir": "demo",
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha", "setMinTime": 10},
        }
        psi = FakeTimePsi([0.1, 0.2, 0.3])

        result = build_time_cube(
            manifest,
            loader=lambda *_args, **_kwargs: [[{"theta": 0.0, "alpha": 0.0}, psi]],
        )

        self.assertEqual(result["setMinTime"], 0)
        self.assertEqual(psi.min_times, [0.0, 0.0, 0.0])

    def test_invalid_slope_becomes_nan_and_bad(self):
        manifest = {
            "subdir": "demo",
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        psi = FakeTimePsi([0.1, None, 0.3])

        result = build_time_cube(
            manifest,
            loader=lambda *_args, **_kwargs: [[{"theta": 0.0, "alpha": 0.0}, psi]],
            set_min_time=0,
        )

        self.assertFalse(result["bad_mask"][0, 0, 0])
        self.assertTrue(result["bad_mask"][1, 0, 0])
        self.assertFalse(result["bad_mask"][2, 0, 0])
        self.assertTrue(math.isnan(result["cube"][1, 0, 0]))

    def test_missing_run_is_bad_for_all_frames(self):
        manifest = {
            "subdir": "demo",
            "sweep": {"theta": [0.0, 1.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        psi = FakeTimePsi([0.1, 0.2])

        def loader(filter_dict, *_args, **_kwargs):
            if filter_dict["theta"] == 1.0:
                return []
            return [[filter_dict, psi]]

        result = build_time_cube(manifest, loader=loader, set_min_time=0)

        self.assertEqual(result["cube"].shape, (2, 2, 1))
        self.assertEqual(result["missing_runs"], 1)
        self.assertTrue(np.all(result["bad_mask"][:, 1, 0]))
        self.assertTrue(np.all(np.isnan(result["cube"][:, 1, 0])))

    def test_discord_fhalf_uses_midpoint_fragment_without_slope_search(self):
        manifest = {
            "subdir": "demo",
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        psi = FakeTimePsi([0.1, 0.2, 0.3])

        result = build_time_cube(
            manifest,
            metric="discord_fhalf",
            loader=lambda *_args, **_kwargs: [[{"theta": 0.0, "alpha": 0.0}, psi]],
            set_min_time=0,
        )

        np.testing.assert_allclose(result["cube"][:, 0, 0], [10.1, 10.2, 10.3])
        self.assertEqual(psi.calls, [])


if __name__ == "__main__":
    unittest.main()
