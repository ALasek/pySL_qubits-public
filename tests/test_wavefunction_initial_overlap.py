import unittest

import cupy as cp

from src.input.default_params import build_default_params
from src.input.wavefunction import wavefunction


class WavefunctionInitialOverlapTests(unittest.TestCase):
    def setUp(self):
        try:
            cp.zeros(1, dtype=cp.float32)
        except (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError) as exc:
            raise unittest.SkipTest(f"CUDA device unavailable: {exc}") from exc

    def test_initial_environment_z_overlaps_stay_on_gpu(self):
        psi = object.__new__(wavefunction)
        psi.nqubitsE = 2
        psi.psi_E_init = cp.array([1, 1, 0, 0], dtype=cp.complex64) / cp.sqrt(cp.float32(2))

        overlaps = psi._initial_environment_z_overlaps()

        self.assertAlmostEqual(overlaps[0], 1.0, places=6)
        self.assertAlmostEqual(overlaps[1], 0.5, places=6)

    def test_pointer_projectors_are_csr_after_buildPsi(self):
        params = build_default_params()
        params.update(
            {
                "nqubits_E": 2,
                "T": 1,
                "printT": 1,
                "QREmaxFragSize": 0,
                "store_correlators": False,
                "seed": 1337,
            }
        )

        psi = wavefunction(params, 0)
        psi.buildPsi()

        self.assertEqual(psi.pointerProj0.format, "csr")
        self.assertEqual(psi.pointerProj1.format, "csr")
        self.assertEqual((psi.pointerProj0 @ cp.ones(8, dtype=cp.complex64)).shape, (4,))

    def test_z_holevo_and_discord_for_bell_state(self):
        params = build_default_params()
        params.update(
            {
                "nqubits_E": 1,
                "T": 1,
                "printT": 1,
                "QREmaxFragSize": 0,
                "store_correlators": False,
                "seed": 1337,
                "clampISE": 0.0,
            }
        )
        psi = wavefunction(params, 0)
        psi.buildPsi()
        psi.psi = cp.zeros(4, dtype=cp.complex64)
        psi.psi[0] = 1 / cp.sqrt(cp.float32(2))
        psi.psi[3] = psi.psi[0]

        psi.print_vn(0, 1)

        expected = float(cp.log(cp.float32(2)).get())
        self.assertAlmostEqual(psi.I_S_Ef_fractionsT[1, 0, 0], 2 * expected, places=5)
        self.assertAlmostEqual(psi.Holevo_Z_S_Ef_fractionsT[1, 0, 0], expected, places=5)
        self.assertAlmostEqual(psi.Discord_Z_S_Ef_fractionsT[1, 0, 0], expected, places=5)

    def test_compute_discord_false_keeps_mi_only_path(self):
        params = build_default_params()
        params.update(
            {
                "nqubits_E": 1,
                "T": 1,
                "printT": 1,
                "QREmaxFragSize": 0,
                "store_correlators": False,
                "compute_discord": False,
                "seed": 1337,
                "clampISE": 0.0,
            }
        )
        psi = wavefunction(params, 0)
        psi.buildPsi()
        psi.psi = cp.zeros(4, dtype=cp.complex64)
        psi.psi[0] = 1 / cp.sqrt(cp.float32(2))
        psi.psi[3] = psi.psi[0]

        psi.print_vn(0, 1)

        expected = float(cp.log(cp.float32(2)).get())
        self.assertAlmostEqual(psi.I_S_Ef_fractionsT[1, 0, 0], 2 * expected, places=5)
        self.assertFalse(hasattr(psi, "Holevo_Z_S_Ef_fractionsT"))
        self.assertFalse(hasattr(psi, "Discord_Z_S_Ef_fractionsT"))


if __name__ == "__main__":
    unittest.main()
