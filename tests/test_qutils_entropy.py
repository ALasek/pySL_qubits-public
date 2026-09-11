import unittest

import cupy as cp
import numpy as np

from src.Qutils import fidelity, partial_trace_cp, trace_distance, vn_entropy_np, von_neumann_entropy_direct


class QutilsEntropyTests(unittest.TestCase):
    def setUp(self):
        try:
            cp.zeros(1, dtype=cp.float32)
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError) as exc:
            raise unittest.SkipTest(f"CUDA device unavailable: {exc}") from exc

    def test_entropy_uses_equivalent_complement_for_large_subsystems(self):
        state = cp.zeros(16, dtype=cp.complex64)
        state[0] = 1 / cp.sqrt(cp.array(2, dtype=cp.float32))
        state[-1] = state[0]

        small = von_neumann_entropy_direct(state, [0], 4)
        large = von_neumann_entropy_direct(state, [1, 2, 3], 4)

        self.assertAlmostEqual(small, large, places=6)

    def test_partial_trace_cp_coerces_numpy_input_to_cupy(self):
        state = cp.array([1, 0, 0, 1], dtype=cp.complex64).get() / cp.sqrt(cp.float32(2)).get()

        rho = partial_trace_cp(state, [0], [2, 2])

        self.assertIsInstance(rho, cp.ndarray)
        np.testing.assert_allclose(rho.get(), np.eye(2) / 2, atol=1e-6)

    def test_numpy_entropy_for_tiny_density_matrix(self):
        rho = np.eye(2) / 2

        entropy = vn_entropy_np(rho)

        self.assertAlmostEqual(entropy, np.log(2), places=7)

    def test_physical_probability_metrics_are_clamped_to_unit_interval(self):
        self.assertEqual(trace_distance(np.array([[2.000001]]), np.zeros((1, 1))), 1.0)
        self.assertEqual(fidelity(np.array([[1.000001]]), np.array([[1.000001]])), 1.0)


if __name__ == "__main__":
    unittest.main()
