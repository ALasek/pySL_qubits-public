import unittest

import cupy as cp
import numpy as np

from src.input.wavefunction import wavefunction


class BranchEnergyDiagnosticsTests(unittest.TestCase):
    def test_pointer_conditioned_storage_energy_gap(self):
        state = wavefunction.__new__(wavefunction)
        state.dim = 4
        state.dimE = 2
        state.nqubitsE = 1
        state.runN = 0
        psi = cp.asarray([1, 0, 0, 1], dtype=cp.complex64) / cp.sqrt(2)
        state.psi_real = cp.real(psi).astype(cp.complex64)
        state.psi_imag = cp.imag(psi).astype(cp.complex64)
        state.branch_probabilities = np.full((2, 1), np.nan)
        state.branch_energy_density = np.full((2, 1), np.nan)
        state.branch_energy_density_std = np.full((2, 1), np.nan)
        state.branch_energy_density_gap = np.full(1, np.nan)
        z = cp.asarray([[1, 0], [0, -1]], dtype=cp.complex64)
        H_store = cp.kron(cp.eye(2, dtype=cp.complex64), z)

        state.record_branch_energy_diagnostics(H_store)

        np.testing.assert_allclose(state.branch_probabilities[:, 0], [0.5, 0.5])
        np.testing.assert_allclose(state.branch_energy_density[:, 0], [-1.0, 1.0])
        np.testing.assert_allclose(state.branch_energy_density_std[:, 0], [0.0, 0.0])
        self.assertAlmostEqual(state.branch_energy_density_gap[0], 2.0)


if __name__ == "__main__":
    unittest.main()
