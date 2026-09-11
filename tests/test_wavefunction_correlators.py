import unittest

import cupy as cp

from src.input.wavefunction import wavefunction


class WavefunctionCorrelatorTests(unittest.TestCase):
    def test_fallback_phi_multiply_uses_selected_source_count(self):
        psi = object.__new__(wavefunction)
        psi._phi_use_loop = True
        phi = cp.asarray([[1.0], [2.0]], dtype=cp.float32)
        H = cp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=cp.float32)

        result = psi._H_phi(H, phi)

        cp.testing.assert_allclose(result, cp.asarray([[2.0], [6.0]], dtype=cp.float32))

    def test_y_axis_matches_pauli_y_sign_convention(self):
        psi = object.__new__(wavefunction)
        psi.dim = 2
        psi.nqubitsE = 1
        psi.dtypef = cp.float32
        psi.dtypec = cp.complex64
        psi.correlator_axis = "Y"
        ket_zero = cp.asarray([1.0, 0.0], dtype=cp.complex64)
        ket_one = cp.asarray([0.0, 1.0], dtype=cp.complex64)

        cp.testing.assert_allclose(
            psi._apply_environment_pauli(ket_zero, 0),
            cp.asarray([0.0, 1.0j], dtype=cp.complex64),
        )
        cp.testing.assert_allclose(
            psi._apply_environment_pauli(ket_one, 0),
            cp.asarray([-1.0j, 0.0], dtype=cp.complex64),
        )


if __name__ == "__main__":
    unittest.main()
