import json
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.analysis.batch_analysis import analyze_batch, compute_metrics_for_psi, save_metric_plots, select_axes


class FakePsi:
    printT = 2

    def __init__(self, slope=0.25, frac=0.5):
        self.slope = slope
        self.frac = frac
        self.nqubitsE = 2
        self.Discord_Z_S_Ef_fractionsT_runAv = np.array(
            [
                [9.0, 9.0, 9.0, 9.0],
                [0.1, 0.2, 0.3, 0.4],
                [8.0, 8.0, 8.0, 8.0],
            ]
        )
        self.corr_spread_radius_T_runAv = np.array([0.1, 0.4, 0.9, 0.2])
        self.corr_offdiag_fraction_T_runAv = np.array([0.0, 0.3, 0.7, 0.2])
        self.corr_total_connected_weight_T_runAv = np.array([1.0, 2.0, 4.0, 1.5])
        self.corr_nn_weight_T_runAv = np.array([0.0, 0.7, 0.1, 0.6])
        self.corr_nn_fraction_T_runAv = np.array([0.0, 0.8, 0.2, 0.5])

    def redundancy_slope_at_T(self, T, cutoff=0.5, minTime=0):
        return [self.slope, 3, 1.0, 1.0, [0]]

    def redundancy_fractionScore_at_T(self, T, cutoff=0.5, minfraction=0.4, minTime=0):
        return [self.frac, 3, 1.0, 1.0]


class BatchAnalysisTests(unittest.TestCase):
    def test_select_axes_uses_manifest_defaults(self):
        manifest = {
            "sweep": {"theta": [0, 1], "alpha": [0, 0.1], "bias": [0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }

        self.assertEqual(select_axes(manifest), ("theta", "alpha"))

    def test_select_axes_rejects_unselected_multi_value_dimension(self):
        manifest = {
            "sweep": {"theta": [0, 1], "alpha": [0, 0.1], "bias": [0, 0.5]},
            "analysis": {"x": "theta", "y": "alpha"},
        }

        with self.assertRaisesRegex(ValueError, "bias"):
            select_axes(manifest)

    def test_select_axes_allows_an_explicit_slice_filter(self):
        manifest = {
            "sweep": {"theta": [0, 1], "alpha": [0, 0.1], "size": [8, 12]},
            "analysis": {"x": "theta", "y": "alpha"},
        }

        self.assertEqual(select_axes(manifest, filters={"size": 8}), ("theta", "alpha"))

    def test_compute_metrics_for_psi(self):
        metrics = compute_metrics_for_psi(FakePsi(), tsample=-1, set_min_time=0)

        self.assertEqual(metrics["qd_slope"], 0.25)
        self.assertEqual(metrics["discord_fhalf"], 0.4)
        self.assertEqual(metrics["redundancy_score"], 2.0)
        self.assertEqual(metrics["best_time"], 6)
        self.assertEqual(metrics["corr_spread_radius"], 0.2)
        self.assertEqual(metrics["corr_total_connected_weight"], 1.5)
        self.assertFalse(metrics["bad"])

    def test_physical_record_metrics_use_holevo_selected_time(self):
        psi = FakePsi()
        psi.printT = 1
        psi.nqubitsE = 4
        psi.Holevo_Z_S_Ef_fractionsT_runAv = np.zeros((5, 5))
        psi.Holevo_Z_S_Ef_fractionsT_runAv[1] = [0.0, 0.2, 0.4, 0.3, 0.1]
        psi.Holevo_Z_S_Ef_fractionsT_runAv[2] = [0.0, 0.55, 0.66, 0.65, 0.2]
        psi.Discord_Z_S_Ef_fractionsT_runAv = np.zeros((5, 5))
        psi.Discord_Z_S_Ef_fractionsT_runAv[2] = [0.5, 0.3, 0.1, 0.12, 0.4]
        psi.rhoS_T = np.repeat((np.eye(2) / 2)[None, :, :], 5, axis=0)
        psi.fragment_quantile_levels = np.array([0.1])
        psi.Holevo_Z_S_Ef_fractionsT_quantiles_runAv = np.zeros((1, 5, 5))
        psi.Holevo_Z_S_Ef_fractionsT_quantiles_runAv[0, 2, 2] = 0.65
        psi.SBS_fid_1 = np.full((4, 5), 0.25)
        psi.SBS_trace_dist_1 = np.array(
            [
                [0.2, 0.8, 1.0, 0.95, 0.4],
                [0.1, 0.7, 0.98, 0.92, 0.3],
                [0.2, 0.6, 0.96, 0.91, 0.2],
                [0.3, 0.5, 0.94, 0.90, 0.1],
            ]
        )
        psi.branch_energy_density_gap_runAv = 1e-5
        psi.branch_energy_density_std_runAv = np.array([0.2, 0.3])

        metrics = compute_metrics_for_psi(psi, tsample=-1, set_min_time=0, holevo_delta=0.1)

        self.assertEqual(metrics["record_time"], 2)
        self.assertEqual(metrics["holevo_fhalf"], 0.66)
        self.assertEqual(metrics["discord_fhalf"], 0.1)
        self.assertEqual(metrics["holevo_redundancy"], 2.0)
        self.assertEqual(metrics["holevo_redundancy_q"], 2.0)
        self.assertEqual(metrics["record_formation_time"], 2)
        self.assertEqual(metrics["record_lifetime"], 1)
        self.assertAlmostEqual(metrics["record_occupancy"], 0.4)
        self.assertAlmostEqual(metrics["kappa_typ_f1"], -0.5 * math.log(0.25))
        self.assertAlmostEqual(metrics["single_site_td_mean"], 0.97)
        self.assertAlmostEqual(metrics["single_site_td_q10"], 0.946)
        self.assertEqual(metrics["single_site_td_lifetime"], 1)
        self.assertAlmostEqual(metrics["single_site_td_occupancy"], 0.4)
        self.assertEqual(metrics["branch_energy_density_gap"], 1e-5)
        self.assertEqual(metrics["branch_energy_density_width"], 0.3)

    def test_record_lifetime_is_elapsed_time_and_trace_distance_is_bounded(self):
        psi = FakePsi()
        psi.printT = 2
        psi.nqubitsE = 2
        psi.Holevo_Z_S_Ef_fractionsT_runAv = np.array(
            [
                np.zeros(31),
                np.full(31, 0.65),
                np.zeros(31),
            ]
        )
        psi.rhoS_T = np.repeat((np.eye(2) / 2)[None, :, :], 31, axis=0)
        psi.SBS_trace_dist_1 = np.array([[1.0001, -0.0001], [1.0002, -0.0002]])

        metrics = compute_metrics_for_psi(psi, tsample=0, set_min_time=10, holevo_delta=0.1)

        self.assertEqual(metrics["record_formation_time"], 10)
        self.assertEqual(metrics["record_lifetime"], 50)
        self.assertEqual(metrics["single_site_td_mean"], 1.0)
        self.assertEqual(metrics["single_site_td_q10"], 1.0)

    def test_missing_pointer_state_fallback_uses_natural_log_units(self):
        psi = FakePsi()
        psi.printT = 1
        psi.nqubitsE = 2
        psi.Holevo_Z_S_Ef_fractionsT_runAv = np.array([[0.0], [0.65], [0.0]])

        metrics = compute_metrics_for_psi(psi, tsample=0, set_min_time=0, holevo_delta=0.1)

        self.assertEqual(metrics["holevo_redundancy"], 2.0)

    def test_explicit_sample_time_controls_correlator_time(self):
        metrics = compute_metrics_for_psi(FakePsi(), tsample=2, set_min_time=0)

        self.assertEqual(metrics["discord_fhalf"], 0.2)
        self.assertEqual(metrics["corr_spread_radius"], 0.4)

    def test_compute_metrics_handles_missing_correlators(self):
        psi = FakePsi()
        del psi.corr_spread_radius_T_runAv

        metrics = compute_metrics_for_psi(psi, tsample=-1, set_min_time=0)

        self.assertTrue(math.isnan(metrics["corr_spread_radius"]))

    def test_analyze_batch_builds_2d_grid_with_exact_filters(self):
        manifest = {
            "subdir": "demo",
            "params_default": {
                "nqubits_E": 15,
                "plot_mode": "none",
                "theta": 999,
            },
            "sweep": {"theta": [0.0, 0.5], "alpha": [0.0, 0.1], "bias": [0.0]},
            "analysis": {"x": "theta", "y": "alpha", "Tsample": -1},
        }
        seen_filters = []

        def loader(filter_dict, subdir, old_data=False):
            seen_filters.append(dict(filter_dict))
            slope = filter_dict["theta"] + filter_dict["alpha"] + 1
            return [[filter_dict, FakePsi(slope=slope, frac=0.5)]]

        result = analyze_batch(manifest, loader=loader)

        self.assertEqual(result["results"]["qd_slope"][0][0], 1.0)
        self.assertEqual(result["results"]["qd_slope"][1][1], 1.6)
        self.assertEqual(
            seen_filters[0],
            {
                "nqubits_E": 15,
                "plot_mode": "none",
                "theta": 0.0,
                "alpha": 0.0,
                "bias": 0.0,
            },
        )

    def test_analyze_batch_marks_missing_runs(self):
        manifest = {
            "subdir": "demo",
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }

        result = analyze_batch(manifest, loader=lambda *_args, **_kwargs: [])

        self.assertTrue(result["results"]["missing"][0][0])
        self.assertTrue(math.isnan(result["results"]["qd_slope"][0][0]))

    def test_analyze_batch_filters_override_fixed_defaults(self):
        manifest = {
            "subdir": "demo",
            "params_default": {"nqubits_E": 13},
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        seen_filters = []

        def loader(filter_dict, subdir, old_data=False):
            seen_filters.append(dict(filter_dict))
            return [[filter_dict, FakePsi()]]

        analyze_batch(manifest, filters={"nqubits_E": 15}, loader=loader)

        self.assertEqual(seen_filters[0]["nqubits_E"], 15)

    def test_save_metric_plots_skips_all_nan_metrics(self):
        result = {
            "manifest": {"subdir": "demo", "params_default": {"nqubits_E": 1, "nqubits_S": 1}},
            "x_axis": "theta",
            "y_axis": None,
            "x_values": [0.0],
            "y_values": None,
            "Tsample": -1,
            "setMinTime": 30,
            "redundancyminfraction": 0.4,
            "filters": {},
            "old_data": False,
            "matched_runs": 1,
            "missing_runs": 0,
            "ambiguous_matches": 0,
            "results": {
                "qd_slope": [[0.2]],
                "qd_slope_bad": [[False]],
                "corr_spread_radius": [[math.nan]],
                "corr_spread_radius_bad": [[True]],
            },
        }

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("src.analysis.batch_analysis._figure_dir", return_value=tmp):
            paths = save_metric_plots(
                result,
                metrics=["qd_slope", "corr_spread_radius"],
                analysis_name="slice_NE-1",
            )

        self.assertEqual(len(paths), 1)
        self.assertIn("qd_slope", paths[0])
        self.assertEqual(os.path.basename(os.path.dirname(paths[0])), "slice_NE-1")

    def test_save_metric_plots_writes_self_contained_bundle_metadata(self):
        manifest = {
            "subdir": "demo",
            "params_default": {
                "nqubits_E": 15,
                "nqubits_S": 1,
                "T": 60,
                "array_param": np.array([1.0, 2.0]),
            },
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha"},
        }
        result = analyze_batch(
            manifest,
            filters={"nqubits_E": 15},
            loader=lambda filter_dict, *_args, **_kwargs: [[filter_dict, FakePsi()]],
        )

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("src.analysis.batch_analysis._figure_dir", return_value=tmp):
            paths = save_metric_plots(result, metrics=["qd_slope"], analysis_name="NE-15")
            bundle_dir = result["analysis_bundle_dir"]

            self.assertEqual(os.path.dirname(paths[0]), bundle_dir)
            for filename in [
                "batch_manifest.json",
                "effective_config.json",
                "analysis_manifest.json",
                "matched_params_summary.json",
                "matched_run_params.json",
                "README.md",
            ]:
                self.assertTrue(os.path.exists(os.path.join(bundle_dir, filename)))

            with open(os.path.join(bundle_dir, "effective_config.json"), encoding="utf-8") as f:
                effective_config = json.load(f)
            with open(os.path.join(bundle_dir, "analysis_manifest.json"), encoding="utf-8") as f:
                analysis_manifest = json.load(f)
            with open(os.path.join(bundle_dir, "matched_params_summary.json"), encoding="utf-8") as f:
                params_summary = json.load(f)
            with open(os.path.join(bundle_dir, "batch_manifest.json"), encoding="utf-8") as f:
                written_manifest = json.load(f)

        self.assertEqual(effective_config["filters"], {"nqubits_E": 15})
        self.assertEqual(analysis_manifest["saved_metrics"], ["qd_slope"])
        self.assertEqual(analysis_manifest["saved_figures"], ["qd_slope.jpg"])
        self.assertEqual(analysis_manifest["figure_files_by_metric"], {"qd_slope": "qd_slope.jpg"})
        self.assertEqual(analysis_manifest["matched_runs"], 1)
        self.assertEqual(params_summary["common"]["nqubits_E"], 15)
        self.assertEqual(written_manifest["params_default"]["array_param"], [1.0, 2.0])

    def test_save_metric_plots_writes_paired_holevo_discord_figure(self):
        result = {
            "manifest": {"subdir": "Paper_B_demo", "params_default": {"nqubits_E": 2}},
            "x_axis": "PaperB_field_strength",
            "y_axis": "PaperB_case",
            "x_values": [0.3, 0.5],
            "y_values": ["zx_uniform_chain", "zx_gaussian_chain"],
            "Tsample": 60,
            "setMinTime": 20,
            "redundancyminfraction": 0.4,
            "holevo_delta": 0.1,
            "holevo_quantile": 0.1,
            "filters": {},
            "old_data": False,
            "matched_runs": 4,
            "missing_runs": 0,
            "ambiguous_matches": 0,
            "results": {
                "holevo_fhalf": [[0.5, 0.6], [0.55, 0.65]],
                "discord_fhalf": [[0.1, 0.05], [0.08, 0.02]],
                "bad": [[False, False], [False, False]],
            },
        }

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("src.analysis.batch_analysis._figure_dir", return_value=tmp):
            paths = save_metric_plots(result, metrics=["holevo_discord_fhalf"])

        self.assertEqual([os.path.basename(path) for path in paths], ["holevo_discord_fhalf.jpg"])

    def test_save_metric_plots_uses_short_bundle_and_figure_names(self):
        manifest = {
            "subdir": "demo",
            "params_default": {
                "nqubits_E": 17,
                "nqubits_S": 1,
                "T": 30,
            },
            "sweep": {"theta": [0.0], "alpha": [0.0]},
            "analysis": {"x": "theta", "y": "alpha", "Tsample": -1},
        }
        result = analyze_batch(
            manifest,
            loader=lambda filter_dict, *_args, **_kwargs: [[filter_dict, FakePsi()]],
        )

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("src.analysis.batch_analysis._figure_dir", return_value=tmp):
            paths = save_metric_plots(
                result,
                metrics=["qd_slope", "redundancy_score", "best_time"],
            )
            bundle_dir = result["analysis_bundle_dir"]

        self.assertEqual(
            [os.path.basename(path) for path in paths],
            ["qd_slope.jpg", "redundancy.jpg", "best_time.jpg"],
        )
        self.assertRegex(
            os.path.basename(bundle_dir),
            r"^N18_NE17_T30_best_\d{8}-\d{6}$",
        )
        for path in paths:
            rel_path = os.path.relpath(path, tmp)
            self.assertLess(len(rel_path), 80)


if __name__ == "__main__":
    unittest.main()
