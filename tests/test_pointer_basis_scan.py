import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.analysis import pointer_basis_scan
from src.analysis.pointer_basis_scan import (
    analyze_pointer_basis_scan,
    branch_matrices_from_psi,
    chi_for_basis_from_blocks,
    compute_blocks,
    make_basis_grid,
    mutual_information_from_branches,
    scan_basis_grid,
)
from src.export import run_store
from src.export.run_store import save_run


class FakePsi:
    I_S_Ef_fractionsT_runAv = np.array([[0.0]])
    I_S_Ef_fractionsT_STD_runAv = np.array([[0.0]])
    printT = 1


def fake_data_dir(root):
    return lambda subdir="": os.path.join(root, subdir) if subdir else root


def basis_index(bits):
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def state_from_terms(n_qubits, terms):
    psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
    for bits, amp in terms:
        psi[basis_index(bits)] = amp
    return psi


def logical_tensor_to_little_endian_flat(tensor):
    return np.transpose(tensor, tuple(reversed(range(tensor.ndim)))).reshape(-1)


class PointerBasisScanTests(unittest.TestCase):
    def scan_for_fragment(self, psi, fragment, n_qubits=3, endianness="big"):
        beta, phi = make_basis_grid(5, 4)
        A0, A1 = branch_matrices_from_psi(
            psi,
            n_qubits,
            system_qubit=0,
            fragment=fragment,
            endianness=endianness,
        )
        G00, G11, G01 = compute_blocks(A0, A1)
        I = mutual_information_from_branches(A0, A1, G00, G11, G01)
        chi_grid = scan_basis_grid(G00, G11, G01, beta, phi)
        return I, chi_grid, G00, G11, G01

    def test_product_state_has_no_record_in_any_basis(self):
        psi = state_from_terms(
            3,
            [
                ((0, 0, 0), 1 / math.sqrt(2)),
                ((1, 0, 0), 1 / math.sqrt(2)),
            ],
        )

        I, chi_grid, *_ = self.scan_for_fragment(psi, fragment=[1])

        self.assertAlmostEqual(I, 0.0, places=12)
        np.testing.assert_allclose(chi_grid, 0.0, atol=1e-12)

    def test_ghz_fragment_has_z_record_and_full_environment_has_discord_remainder(self):
        psi = state_from_terms(
            4,
            [
                ((0, 0, 0, 0), 1 / math.sqrt(2)),
                ((1, 1, 1, 1), 1 / math.sqrt(2)),
            ],
        )

        I_fragment, chi_grid, *_ = self.scan_for_fragment(psi, fragment=[1], n_qubits=4)
        self.assertAlmostEqual(I_fragment, 1.0, places=12)
        self.assertAlmostEqual(chi_grid[0, 0], 1.0, places=12)
        self.assertAlmostEqual(np.nanmax(chi_grid), 1.0, places=12)

        I_full, chi_full, *_ = self.scan_for_fragment(psi, fragment=[1, 2, 3], n_qubits=4)
        self.assertAlmostEqual(I_full, 2.0, places=12)
        self.assertAlmostEqual(chi_full[0, 0], 1.0, places=12)
        self.assertAlmostEqual(I_full - chi_full[0, 0], 1.0, places=12)

    def test_bell_record_depends_on_fragment_membership(self):
        psi = state_from_terms(
            3,
            [
                ((0, 0, 0), 1 / math.sqrt(2)),
                ((1, 1, 0), 1 / math.sqrt(2)),
            ],
        )

        I_record, chi_record, *_ = self.scan_for_fragment(psi, fragment=[1])
        I_irrelevant, chi_irrelevant, *_ = self.scan_for_fragment(psi, fragment=[2])

        self.assertAlmostEqual(I_record, 2.0, places=12)
        self.assertAlmostEqual(chi_record[0, 0], 1.0, places=12)
        self.assertAlmostEqual(I_irrelevant, 0.0, places=12)
        np.testing.assert_allclose(chi_irrelevant, 0.0, atol=1e-12)

    def test_little_endian_flattening_preserves_logical_qubit_labels(self):
        tensor = np.zeros((2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0] = 1 / math.sqrt(2)
        tensor[1, 0, 1] = 1 / math.sqrt(2)
        psi_little = logical_tensor_to_little_endian_flat(tensor)

        I_record, chi_record, *_ = self.scan_for_fragment(
            psi_little,
            fragment=[2],
            n_qubits=3,
            endianness="little",
        )
        I_irrelevant, chi_irrelevant, *_ = self.scan_for_fragment(
            psi_little,
            fragment=[1],
            n_qubits=3,
            endianness="little",
        )

        self.assertAlmostEqual(I_record, 2.0, places=12)
        self.assertAlmostEqual(chi_record[0, 0], 1.0, places=12)
        self.assertAlmostEqual(I_irrelevant, 0.0, places=12)
        np.testing.assert_allclose(chi_irrelevant, 0.0, atol=1e-12)

    def test_beta_zero_and_pi_are_relabelings_for_holevo(self):
        psi = state_from_terms(
            3,
            [
                ((0, 0, 0), 1 / math.sqrt(2)),
                ((1, 1, 0), 1 / math.sqrt(2)),
            ],
        )
        _, _, G00, G11, G01 = self.scan_for_fragment(psi, fragment=[1])

        chi_zero = chi_for_basis_from_blocks(G00, G11, G01, beta=0.0, phi=0.0)
        chi_pi = chi_for_basis_from_blocks(G00, G11, G01, beta=math.pi, phi=0.0)

        self.assertAlmostEqual(chi_zero, chi_pi, places=12)

    def test_analyzer_loads_pysl_v2_snapshot_realization(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)), \
                    patch.object(pointer_basis_scan, "get_data_dir", fake_data_dir(tmp)):
                psi = state_from_terms(
                    3,
                    [
                        ((0, 0, 0), 1 / math.sqrt(2)),
                        ((1, 1, 0), 1 / math.sqrt(2)),
                    ],
                )
                snapshots = np.zeros((1, 2, psi.size), dtype=np.float64)
                snapshots[0, 0, :] = psi.real
                snapshots[0, 1, :] = psi.imag
                source_path = os.path.join(tmp, "snapshot.npy")
                np.save(source_path, snapshots)
                fake_psi = FakePsi()
                fake_psi.psi_snapshot_records = [
                    {
                        "run_index": 0,
                        "seed": 1337,
                        "path": source_path,
                        "shape": list(snapshots.shape),
                        "dtype": "float64",
                        "sample_times": [0.0],
                    }
                ]
                save_result = save_run(
                    fake_psi,
                    {"nqubits_S": 1, "nqubits_E": 2, "theta": 0.1},
                    subdir="demo",
                    write_legacy=False,
                )

                out_dir = os.path.join(tmp, "out")
                summary = analyze_pointer_basis_scan(
                    subdir="demo",
                    out_dir=out_dir,
                    run_ids=[save_result["run_id"]],
                    fragment_sizes=[1],
                    fragment_mode="explicit",
                    explicit_fragments={"1": [[1]]},
                    beta_count=5,
                    phi_count=4,
                    device="cpu",
                    dtype="complex128",
                    plot=False,
                )
                checkpoint_dir = os.path.join(out_dir, "checkpoints")
                self.assertEqual(len(os.listdir(checkpoint_dir)), 1)

                with patch.object(pointer_basis_scan, "scan_basis_grid", side_effect=AssertionError("cache miss")):
                    cached_summary = analyze_pointer_basis_scan(
                        subdir="demo",
                        out_dir=out_dir,
                        run_ids=[save_result["run_id"]],
                        fragment_sizes=[1],
                        fragment_mode="explicit",
                        explicit_fragments={"1": [[1]]},
                        beta_count=5,
                        phi_count=4,
                        device="cpu",
                        dtype="complex128",
                        plot=False,
                    )

        result = summary["results_by_fragment_size"]["1"]
        self.assertEqual(summary["num_realizations"], 1)
        self.assertAlmostEqual(result["mean_chi_z"], 1.0, places=12)
        self.assertAlmostEqual(result["mean_I"], 2.0, places=12)
        self.assertAlmostEqual(result["delta_chi_global"], 0.0, places=12)
        self.assertEqual(cached_summary["checkpoint"]["hits"], 1)
        self.assertEqual(cached_summary["checkpoint"]["writes"], 0)

    def test_checkpoint_is_invalidated_when_same_run_id_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_store, "get_data_dir", fake_data_dir(tmp)), \
                    patch.object(pointer_basis_scan, "get_data_dir", fake_data_dir(tmp)):
                params = {"nqubits_S": 1, "nqubits_E": 2, "theta": 0.1}
                states = [
                    state_from_terms(
                        3,
                        [
                            ((0, 0, 0), 1 / math.sqrt(2)),
                            ((1, 1, 0), 1 / math.sqrt(2)),
                        ],
                    ),
                    state_from_terms(
                        3,
                        [
                            ((0, 0, 0), 1 / math.sqrt(2)),
                            ((1, 0, 0), 1 / math.sqrt(2)),
                        ],
                    ),
                ]
                out_dir = os.path.join(tmp, "out")
                summaries = []

                for index, psi in enumerate(states):
                    snapshots = np.zeros((1, 2, psi.size), dtype=np.float64)
                    snapshots[0, 0, :] = psi.real
                    snapshots[0, 1, :] = psi.imag
                    source_path = os.path.join(tmp, f"snapshot_{index}.npy")
                    np.save(source_path, snapshots)
                    fake_psi = FakePsi()
                    fake_psi.psi_snapshot_records = [
                        {
                            "run_index": 0,
                            "seed": 1337,
                            "path": source_path,
                            "shape": list(snapshots.shape),
                            "dtype": "float64",
                            "sample_times": [0.0],
                        }
                    ]
                    save_result = save_run(fake_psi, params, subdir="demo", write_legacy=False)
                    summaries.append(
                        analyze_pointer_basis_scan(
                            subdir="demo",
                            out_dir=out_dir,
                            run_ids=[save_result["run_id"]],
                            fragment_sizes=[1],
                            fragment_mode="explicit",
                            explicit_fragments={"1": [[1]]},
                            beta_count=5,
                            phi_count=4,
                            device="cpu",
                            dtype="complex128",
                            plot=False,
                        )
                    )

        first = summaries[0]["results_by_fragment_size"]["1"]
        second = summaries[1]["results_by_fragment_size"]["1"]
        self.assertAlmostEqual(first["mean_chi_z"], 1.0, places=12)
        self.assertAlmostEqual(second["mean_chi_z"], 0.0, places=12)
        self.assertEqual(summaries[1]["checkpoint"]["hits"], 0)
        self.assertEqual(summaries[1]["checkpoint"]["writes"], 1)


if __name__ == "__main__":
    unittest.main()
