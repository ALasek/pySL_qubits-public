import unittest

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as spsp

from src.Qutils import (
    apply_one_qubit_gate_cp,
    apply_two_qubit_gate,
    apply_two_qubit_gate_cp,
    embedded_two_qubit_entries_np,
)
from src.input.wavefunction import wavefunction
from src.sparse_operator import HybridSparseOperator, SumSparseOperator


class SparseEmbeddingTests(unittest.TestCase):
    def setUp(self):
        try:
            cp.zeros(1, dtype=cp.float32)
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError) as exc:
            raise unittest.SkipTest(f"CUDA device unavailable: {exc}") from exc

    def test_two_qubit_embedding_matches_dense_nonadjacent_operator(self):
        G = cp.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 2.0, 0.25j, 0.0],
                [0.0, -0.25j, 3.0, 0.0],
                [0.5, 0.0, 0.0, 4.0],
            ],
            dtype=cp.complex64,
        )

        sparse = apply_two_qubit_gate_cp(G, 0, 2, 3).toarray().get()
        dense = apply_two_qubit_gate(G.get(), 0, 2, 3)

        np.testing.assert_allclose(sparse, dense, atol=1e-6)

    def test_cpu_two_qubit_embedding_matches_dense_nonadjacent_operator(self):
        G = cp.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 2.0, 0.25j, 0.0],
                [0.0, -0.25j, 3.0, 0.0],
                [0.5, 0.0, 0.0, 4.0],
            ],
            dtype=cp.complex64,
        )

        rows, cols, data = embedded_two_qubit_entries_np(G, 0, 2, 3, dtypecp=cp.complex64)
        sparse = spsp.coo_matrix((data, (rows, cols)), shape=(8, 8)).toarray()
        dense = apply_two_qubit_gate(G.get(), 0, 2, 3)

        np.testing.assert_allclose(sparse, dense, atol=1e-6)

    def test_one_qubit_embedding_matches_dense_operator(self):
        G = cp.array([[0.2, 1.0j], [-1.0j, -0.4]], dtype=cp.complex64)

        sparse = apply_one_qubit_gate_cp(G, 1, 3).toarray().get()
        dense = np.kron(np.eye(2), np.kron(G.get(), np.eye(2)))

        np.testing.assert_allclose(sparse, dense, atol=1e-6)

    def test_precompute_sums_duplicate_sparse_entries_once(self):
        psi = object.__new__(wavefunction)
        psi.bundle = 1
        psi.nqubits = 2
        psi.nqubitsS = 1
        psi.nqubitsE = 1
        psi.dim = 4
        psi.dtypec = cp.complex64
        psi.params = {"H_precompute_mode": "sum_terms"}

        H = type("DummyHamiltonian", (), {})()
        H.bondsSE = [[0, 1], [0, 1]]
        H.J_SE = [1.0, -1.0]
        H.H_SE_matrix = cp.eye(4, dtype=cp.complex64)
        H.J_E = []
        H.bondsEE = []
        H.J_EE = []

        psi.precomputeH_cpspCombine(H)

        np.testing.assert_allclose(psi.H_all.toarray().get(), np.zeros((4, 4)), atol=1e-6)

    def test_precompute_combined_cpu_returns_single_csr(self):
        psi = object.__new__(wavefunction)
        psi.bundle = 1
        psi.nqubits = 2
        psi.nqubitsS = 1
        psi.nqubitsE = 1
        psi.dim = 4
        psi.dtypec = cp.complex64
        psi.params = {"H_precompute_mode": "combined_cpu"}

        H = type("DummyHamiltonian", (), {})()
        H.bondsSE = [[0, 1], [0, 1]]
        H.J_SE = [1.0, -1.0]
        H.H_SE_matrix = cp.eye(4, dtype=cp.complex64)
        H.J_E = []
        H.bondsEE = []
        H.J_EE = []

        psi.precomputeH_cpspCombine(H)

        self.assertIsInstance(psi.H_all, cpsp.csr_matrix)
        np.testing.assert_allclose(psi.H_all.toarray().get(), np.zeros((4, 4)), atol=1e-6)

    def test_precompute_grouped_cpu_extracts_diagonal_terms(self):
        psi = object.__new__(wavefunction)
        psi.bundle = 1
        psi.nqubits = 2
        psi.nqubitsS = 1
        psi.nqubitsE = 1
        psi.dim = 4
        psi.dtypec = cp.complex64
        psi.params = {"H_precompute_mode": "grouped_cpu", "H_precompute_group_size": 2}

        H = type("DummyHamiltonian", (), {})()
        H.bondsSE = [[0, 1]]
        H.J_SE = [0.5]
        H.H_SE_matrix = cp.diag(cp.array([1, 2, 3, 4], dtype=cp.complex64))
        H.J_E = []
        H.bondsEE = []
        H.J_EE = []

        psi.precomputeH_cpspCombine(H)

        self.assertIsInstance(psi.H_all, HybridSparseOperator)
        self.assertEqual(len(psi.H_all.terms), 0)
        expected = 0.5 * H.H_SE_matrix
        np.testing.assert_allclose(psi.H_all.toarray().get(), expected.get(), atol=1e-6)

    def test_precompute_grouped_cpu_combines_sparse_groups(self):
        psi = object.__new__(wavefunction)
        psi.bundle = 1
        psi.nqubits = 2
        psi.nqubitsS = 1
        psi.nqubitsE = 1
        psi.dim = 4
        psi.dtypec = cp.complex64
        psi.params = {"H_precompute_mode": "grouped_cpu", "H_precompute_group_size": 1}

        H = type("DummyHamiltonian", (), {})()
        H.bondsSE = [[0, 1]]
        H.J_SE = [1.0]
        H.H_SE_matrix = cp.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 2, 0],
            ],
            dtype=cp.complex64,
        )
        H.J_E = []
        H.bondsEE = []
        H.J_EE = []

        psi.precomputeH_cpspCombine(H)

        self.assertIsInstance(psi.H_all, HybridSparseOperator)
        self.assertEqual(len(psi.H_all.terms), 1)
        np.testing.assert_allclose(psi.H_all.toarray().get(), H.H_SE_matrix.get(), atol=1e-6)

    def test_precompute_grouped_cpu_splits_mixed_local_terms(self):
        psi = object.__new__(wavefunction)
        psi.bundle = 1
        psi.nqubits = 2
        psi.nqubitsS = 1
        psi.nqubitsE = 1
        psi.dim = 4
        psi.dtypec = cp.complex64
        psi.params = {"H_precompute_mode": "grouped_cpu", "H_precompute_group_size": 2}

        H = type("DummyHamiltonian", (), {})()
        H.bondsSE = [[0, 1]]
        H.J_SE = [1.0]
        H.H_SE_matrix = cp.array(
            [
                [1, 0.5, 0, 0],
                [0.5, 2, 0, 0],
                [0, 0, 3, 0.25j],
                [0, 0, -0.25j, 4],
            ],
            dtype=cp.complex64,
        )
        H.J_E = []
        H.bondsEE = []
        H.J_EE = []

        psi.precomputeH_cpspCombine(H)

        self.assertIsInstance(psi.H_all, HybridSparseOperator)
        self.assertIsNotNone(psi.H_all.diagonal)
        self.assertEqual(len(psi.H_all.terms), 1)
        np.testing.assert_allclose(psi.H_all.toarray().get(), H.H_SE_matrix.get(), atol=1e-6)

    def test_sum_sparse_operator_matmul_and_add(self):
        first = apply_one_qubit_gate_cp(cp.array([[1, 0], [0, -1]], dtype=cp.complex64), 0, 2)
        second = apply_one_qubit_gate_cp(cp.array([[0, 1], [1, 0]], dtype=cp.complex64), 1, 2)
        op = SumSparseOperator([first], shape=(4, 4), dtype=cp.complex64) + second
        vector = cp.array([1, 2, 3, 4], dtype=cp.complex64)

        got = op @ vector
        expected = (first + second) @ vector

        np.testing.assert_allclose(got.get(), expected.get(), atol=1e-6)

    def test_hybrid_sparse_operator_matmul_and_add(self):
        term = apply_one_qubit_gate_cp(cp.array([[0, 1], [1, 0]], dtype=cp.complex64), 1, 2)
        diagonal = cp.array([1, 2, 3, 4], dtype=cp.complex64)
        op = HybridSparseOperator(diagonal, [term], shape=(4, 4), dtype=cp.complex64)
        vector = cp.array([1, 2, 3, 4], dtype=cp.complex64)

        got = op @ vector
        expected = (cp.diag(diagonal) + term.toarray()) @ vector

        np.testing.assert_allclose(got.get(), expected.get(), atol=1e-6)

        matrix = cp.stack([vector, 2 * vector], axis=1)
        got_matrix = op @ matrix
        expected_matrix = (cp.diag(diagonal) + term.toarray()) @ matrix
        np.testing.assert_allclose(got_matrix.get(), expected_matrix.get(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
