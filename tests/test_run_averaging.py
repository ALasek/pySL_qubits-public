import unittest

import numpy as np

from src.analysis.correlator_metrics import CORRELATOR_METRIC_NAMES
from src.input.wavefunction import wavefunction


class RunAveragingTests(unittest.TestCase):
    def test_do_run_averages_propagates_fragment_and_run_errors(self):
        psi = object.__new__(wavefunction)
        psi.AverageOverRunsN = 2
        psi.I_S_Ef_fractionsT = np.array([[[1.0, 3.0]], [[2.0, 4.0]]])
        psi.I_S_Ef_fractionsT_STD = np.array([[[0.2, 0.4]], [[0.0, 0.0]]])
        psi.I_S_Ef_fractionsT_Nsamples = np.array([[[4, 4]], [[1, 1]]])
        psi.S_vn = np.array([[0.2, 0.6]])
        psi.TraceDist_fractionsT = np.zeros_like(psi.I_S_Ef_fractionsT)
        psi.TraceDist_fractionsT_STD = np.zeros_like(psi.I_S_Ef_fractionsT)
        psi.Holevo_Z_S_Ef_fractionsT = np.array([[[0.5, 1.5]], [[1.0, 2.0]]])
        psi.Holevo_Z_S_Ef_fractionsT_STD = np.array([[[0.1, 0.3]], [[0.0, 0.0]]])
        psi.Holevo_Z_S_Ef_fractionsT_Nsamples = np.array([[[4, 4]], [[1, 1]]])
        psi.Discord_Z_S_Ef_fractionsT = np.array([[[0.5, 1.5]], [[1.0, 2.0]]])
        psi.Discord_Z_S_Ef_fractionsT_STD = np.array([[[0.1, 0.3]], [[0.0, 0.0]]])
        psi.Discord_Z_S_Ef_fractionsT_Nsamples = np.array([[[4, 4]], [[1, 1]]])
        psi._metric_runs_finalized = 0
        psi.store_correlators = False

        psi.doRunAvs()

        np.testing.assert_allclose(psi.I_S_Ef_fractionsT_runAv[:, 0], [2.0, 3.0])
        np.testing.assert_allclose(psi.I_S_Ef_fractionsT_runSEM[:, 0], [1.0, 1.0])
        expected_fragment_std = np.sqrt(0.2**2 + 0.4**2) / 2
        expected_fragment_sem = np.sqrt((0.2**2 / 4) + (0.4**2 / 4)) / 2
        self.assertAlmostEqual(psi.I_S_Ef_fractionsT_fragmentSTD_runAv[0, 0], 0.3)
        self.assertAlmostEqual(psi.I_S_Ef_fractionsT_fragmentSTD_prop[0, 0], expected_fragment_std)
        self.assertAlmostEqual(psi.I_S_Ef_fractionsT_fragmentSEM_runAv[0, 0], expected_fragment_sem)
        self.assertAlmostEqual(
            psi.I_S_Ef_fractionsT_STD_runAv[0, 0],
            np.sqrt(1.0 + expected_fragment_std**2),
        )
        self.assertAlmostEqual(psi.S_vn_runAv[0], 0.4)
        self.assertAlmostEqual(psi.S_vn_runSEM[0], 0.2)
        np.testing.assert_allclose(psi.Holevo_Z_S_Ef_fractionsT_runAv[:, 0], [1.0, 1.5])
        np.testing.assert_allclose(psi.Discord_Z_S_Ef_fractionsT_runAv[:, 0], [1.0, 1.5])
        self.assertAlmostEqual(psi.Holevo_Z_S_Ef_fractionsT_fragmentSTD_runAv[0, 0], 0.2)
        self.assertAlmostEqual(psi.Discord_Z_S_Ef_fractionsT_fragmentSTD_runAv[0, 0], 0.2)

    def test_correlator_plots_use_mean_magnitude_over_runs(self):
        psi = object.__new__(wavefunction)
        psi.AverageOverRunsN = 2
        psi.QREmaxFragSize = 0
        psi.store_correlators = True
        psi._metric_runs_finalized = 0
        psi._run_metric_sums = {}
        psi._run_metric_counts = {}
        for name in CORRELATOR_METRIC_NAMES:
            setattr(psi, f"{name}_T", np.full([1, 2], np.nan))

        for run_idx, sign in enumerate([1.0, -1.0]):
            psi.runN = run_idx
            psi.rhoS_T = [np.eye(2) / 2]
            psi.rhoS_purity = [0.5]
            psi.SBS_qre = [None]
            psi.SBS_trace_dist = [None]
            psi.SBS_fid = [None]
            psi.C_correlators = np.array([[[sign + 0j]]], dtype=np.complex64)
            psi.C_connected_correlators = np.array([[[sign + 0j]]], dtype=np.complex64)
            psi.C_SE_correlators = np.array([[[2 * sign + 0j]]], dtype=np.complex64)
            psi.finish_run_metrics()

        psi._finalize_run_metrics()

        np.testing.assert_allclose(psi.C_correlators, [[[0.0 + 0.0j]]])
        np.testing.assert_allclose(psi.C_connected_correlators, [[[0.0 + 0.0j]]])
        np.testing.assert_allclose(psi.C_SE_correlators, [[[0.0 + 0.0j]]])
        np.testing.assert_allclose(psi.C_correlators_abs_runAv, [[[1.0]]])
        np.testing.assert_allclose(psi.C_connected_correlators_abs_runAv, [[[1.0]]])
        np.testing.assert_allclose(psi.C_SE_correlators_abs_runAv, [[[2.0]]])


if __name__ == "__main__":
    unittest.main()
