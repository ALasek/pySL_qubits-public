import unittest

import numpy as np
from scipy.linalg import expm

from src.input.wavefunction import wavefunction


def _state_for_step(psi):
    state = wavefunction.__new__(wavefunction)
    state.psi_real = np.real(psi).astype(np.complex128)
    state.psi_imag = np.imag(psi).astype(np.complex128)
    state.store_correlators = False
    state._profile_current_evolve_step = False
    return state


def _propagate(psi, hamiltonian, duration, steps):
    state = _state_for_step(psi)
    dt = duration / steps
    for _ in range(steps):
        state._evolve_staggered(hamiltonian, dt)
    return state.psi_real + 1j * state.psi_imag


class LeapfrogTests(unittest.TestCase):
    def setUp(self):
        self.psi = np.array([1.0, 0.3 + 0.4j], dtype=np.complex128)
        self.psi /= np.linalg.norm(self.psi)
        self.hamiltonians = {
            "real": np.array([[0.3, 0.7], [0.7, -0.2]], dtype=np.complex128),
            "complex": np.array([[0.3, 0.2 - 0.8j], [0.2 + 0.8j, -0.2]], dtype=np.complex128),
        }

    def test_step_is_time_reversible_for_real_and_complex_hamiltonians(self):
        for name, hamiltonian in self.hamiltonians.items():
            with self.subTest(name=name):
                state = _state_for_step(self.psi)
                initial_real = state.psi_real.copy()
                initial_imag = state.psi_imag.copy()

                state._evolve_staggered(hamiltonian, 0.1)
                state._evolve_staggered(hamiltonian, -0.1)

                np.testing.assert_allclose(state.psi_real, initial_real, atol=1e-14, rtol=0)
                np.testing.assert_allclose(state.psi_imag, initial_imag, atol=1e-14, rtol=0)

    def test_global_error_is_second_order_for_real_and_complex_hamiltonians(self):
        duration = 1.0
        for name, hamiltonian in self.hamiltonians.items():
            with self.subTest(name=name):
                exact = expm(-1j * hamiltonian * duration) @ self.psi
                errors = [
                    np.linalg.norm(_propagate(self.psi, hamiltonian, duration, steps) - exact)
                    for steps in (10, 20, 40)
                ]

                self.assertGreater(errors[0] / errors[1], 3.8)
                self.assertGreater(errors[1] / errors[2], 3.8)

    def test_modified_norm_is_invariant(self):
        hamiltonian = self.hamiltonians["complex"]
        state = _state_for_step(self.psi)
        dt = 0.05
        initial = state.modified_staggered_norm(hamiltonian, dt)

        for _ in range(100):
            state._evolve_staggered(hamiltonian, dt)

        self.assertAlmostEqual(state.modified_staggered_norm(hamiltonian, dt), initial, places=12)

    def test_correlator_states_use_the_same_second_order_step(self):
        hamiltonian = self.hamiltonians["complex"]
        phi = np.column_stack((self.psi, np.array([0.4 - 0.2j, 0.7 + 0.1j])))
        duration = 1.0
        exact = expm(-1j * hamiltonian * duration) @ phi
        errors = []

        for steps in (10, 20, 40):
            state = _state_for_step(self.psi)
            state.store_correlators = True
            state.phi_real = np.real(phi).astype(np.complex128)
            state.phi_imag = np.imag(phi).astype(np.complex128)
            state.nqubitsE = phi.shape[1]
            state._phi_diag_done = True
            state._perf = {}
            dt = duration / steps
            for _ in range(steps):
                state._evolve_staggered(hamiltonian, dt)
            errors.append(np.linalg.norm(state.phi_real + 1j * state.phi_imag - exact))

        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertGreater(errors[1] / errors[2], 3.8)


if __name__ == "__main__":
    unittest.main()
