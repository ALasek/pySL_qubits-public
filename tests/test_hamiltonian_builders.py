import unittest

import cupy as cp
import numpy as np

from batch_configs.defaults import build_batch_params_default
from eigensolver.sz_eigensolver import check_sz_conservation
from src.input.hamiltonian import hamiltonian


def mironowicz_params(model):
    params = build_batch_params_default()
    params.update(
        {
            "H_SE_Special": model,
            "Mironowicz_alpha2": 0.84,
            "Mironowicz_theta": np.pi / 2,
            "H_EE_J": 0.04,
            "H_EE_p": "Const",
            "nqubits_E": 3,
            "seed": 1337,
        }
    )
    return params


class HamiltonianBuilderTests(unittest.TestCase):
    def test_generic_explicit_bonds_receive_constant_couplings(self):
        params = mironowicz_params(0)
        params.update(
            {
                "H_SE_bonds": [[0, 1], [0, 3]],
                "H_SE_J": 0.2,
                "H_SE_p": "Const",
                "H_EE_bonds": [[1, 3]],
                "H_EE_J": 0.4,
                "H_EE_p": "Const",
            }
        )

        H = hamiltonian(params, seed=params["seed"])

        self.assertEqual(H.bondsSE, [[0, 1], [0, 3]])
        self.assertEqual(H.J_SE, [0.2, 0.2])
        self.assertEqual(H.bondsEE, [[1, 3]])
        self.assertEqual(H.J_EE, [0.4])

    def test_transverse_explicit_ee_bonds_receive_couplings(self):
        params = mironowicz_params("Mironowicz_rand_EE_transverse")
        params["H_EE_bonds"] = [[1, 3]]

        H = hamiltonian(params, seed=params["seed"])

        self.assertEqual(H.bondsEE, [[1, 3]])
        self.assertEqual(H.J_EE, [params["H_EE_J"]])

    def test_mironowicz_rand_random_fields_are_reproducible(self):
        params = mironowicz_params("Mironowicz_rand")
        params["nqubits_E"] = 8
        params["H_SE_J"] = 0.1
        params["Mironowicz_alpha2"] = 0.7

        first = hamiltonian(params, seed=params["seed"])
        second = hamiltonian(params, seed=params["seed"])

        np.testing.assert_allclose(second.J_SE, first.J_SE)
        np.testing.assert_allclose(second.J_E, first.J_E)

    def test_mironowicz_h0_offsets_random_fields_without_changing_random_draws(self):
        zero = mironowicz_params("Mironowicz_rand")
        zero["nqubits_E"] = 8
        zero["Mironowicz_h0"] = 0.0
        shifted = dict(zero)
        shifted["Mironowicz_h0"] = -0.23

        H_zero = hamiltonian(zero, seed=zero["seed"])
        H_shifted = hamiltonian(shifted, seed=shifted["seed"])

        np.testing.assert_allclose(H_shifted.J_SE, H_zero.J_SE)
        np.testing.assert_allclose(np.asarray(H_shifted.J_E), np.asarray(H_zero.J_E) - 0.23)

    def test_mironowicz_h0_zero_is_backward_compatible_when_key_is_omitted(self):
        omitted = mironowicz_params("Mironowicz_rand")
        omitted.pop("Mironowicz_h0", None)
        explicit = dict(omitted)
        explicit["Mironowicz_h0"] = 0.0

        H_omitted = hamiltonian(omitted, seed=omitted["seed"])
        H_explicit = hamiltonian(explicit, seed=explicit["seed"])

        np.testing.assert_allclose(H_explicit.J_SE, H_omitted.J_SE)
        np.testing.assert_allclose(H_explicit.J_E, H_omitted.J_E)

    def test_mironowicz_rand_se_and_e_random_fields_are_independent_streams(self):
        params = mironowicz_params("Mironowicz_rand")
        params["nqubits_E"] = 8
        params["H_SE_J"] = 0.1
        params["Mironowicz_alpha2"] = 0.7

        H = hamiltonian(params, seed=params["seed"])
        normalized_se = np.asarray(H.J_SE, dtype=float) / params["H_SE_J"]
        normalized_e = np.asarray(H.J_E, dtype=float) / params["Mironowicz_alpha2"]

        self.assertFalse(np.allclose(normalized_se, normalized_e))

    def test_mironowicz_epsilon_variants_keep_same_random_fields(self):
        params_standard = mironowicz_params("Mironowicz_rand")
        params_epsilon = mironowicz_params("Mironowicz_rand_epsilon")
        params_mod = mironowicz_params("Mironowicz_rand_mod_epsilon")

        H_standard = hamiltonian(params_standard, seed=params_standard["seed"])
        H_epsilon = hamiltonian(params_epsilon, seed=params_epsilon["seed"])
        H_mod = hamiltonian(params_mod, seed=params_mod["seed"])

        np.testing.assert_allclose(H_epsilon.J_SE, H_standard.J_SE)
        np.testing.assert_allclose(H_epsilon.J_E, H_standard.J_E)
        np.testing.assert_allclose(H_mod.J_SE, H_standard.J_SE)
        np.testing.assert_allclose(H_mod.J_E, H_standard.J_E)

    def test_random_ee_couplings_are_reproducible_across_ee_variants(self):
        params_standard = mironowicz_params("Mironowicz_rand_EE")
        params_transverse = mironowicz_params("Mironowicz_rand_EE_transverse")
        params_standard["H_EE_p"] = "Norm"
        params_transverse["H_EE_p"] = "Norm"

        H_standard = hamiltonian(params_standard, seed=params_standard["seed"])
        H_transverse = hamiltonian(params_transverse, seed=params_transverse["seed"])

        np.testing.assert_allclose(H_transverse.J_EE, H_standard.J_EE)

    def test_mironowicz_rand_ee_transverse_uses_explicit_ee_matrix(self):
        params = mironowicz_params("Mironowicz_rand_EE_transverse")
        H = hamiltonian(params, seed=params["seed"])

        i = cp.array([[1, 0], [0, 1]], dtype=H.dtypec)
        x = cp.array([[0, 1], [1, 0]], dtype=H.dtypec)
        y = cp.array([[0, -1j], [1j, 0]], dtype=H.dtypec)
        z = cp.array([[1, 0], [0, -1]], dtype=H.dtypec)
        expected = 0.5 * (cp.kron(z, i) + cp.kron(z, x) + cp.kron(z, y))

        np.testing.assert_allclose(cp.asnumpy(H.H_EE_matrix), cp.asnumpy(expected))
        self.assertEqual(H.bondsEE, [[1, 2], [2, 3]])
        self.assertEqual(H.J_EE, [0.04, 0.04])

    def test_mironowicz_rand_ee_transverse_keeps_same_random_fields(self):
        params_standard = mironowicz_params("Mironowicz_rand_EE")
        params_transverse = mironowicz_params("Mironowicz_rand_EE_transverse")

        H_standard = hamiltonian(params_standard, seed=params_standard["seed"])
        H_transverse = hamiltonian(params_transverse, seed=params_transverse["seed"])

        np.testing.assert_allclose(H_transverse.J_SE, H_standard.J_SE)
        np.testing.assert_allclose(H_transverse.J_E, H_standard.J_E)

    def test_mironowicz_rand_ee_zx_keeps_only_zx_ee_matrix(self):
        params = mironowicz_params("Mironowicz_rand_EE_ZX")
        H = hamiltonian(params, seed=params["seed"])

        x = cp.array([[0, 1], [1, 0]], dtype=H.dtypec)
        z = cp.array([[1, 0], [0, -1]], dtype=H.dtypec)
        expected = cp.kron(z, x)

        np.testing.assert_allclose(cp.asnumpy(H.H_EE_matrix), cp.asnumpy(expected))
        self.assertEqual(H.bondsEE, [[1, 2], [2, 3]])
        self.assertEqual(H.J_EE, [0.04, 0.04])

    def test_mironowicz_rand_ee_zx_keeps_same_random_fields(self):
        params_standard = mironowicz_params("Mironowicz_rand_EE")
        params_zx = mironowicz_params("Mironowicz_rand_EE_ZX")

        H_standard = hamiltonian(params_standard, seed=params_standard["seed"])
        H_zx = hamiltonian(params_zx, seed=params_zx["seed"])

        np.testing.assert_allclose(H_zx.J_SE, H_standard.J_SE)
        np.testing.assert_allclose(H_zx.J_E, H_standard.J_E)

    def test_mironowicz_rand_ee_zz_to_zx_rotates_ee_matrix(self):
        params = mironowicz_params("Mironowicz_rand_EE_ZZ_to_ZX")
        params["Mironowicz_ZZ_to_ZX_epsilon"] = 0.37
        H = hamiltonian(params, seed=params["seed"])

        x = cp.array([[0, 1], [1, 0]], dtype=H.dtypec)
        z = cp.array([[1, 0], [0, -1]], dtype=H.dtypec)
        epsilon = params["Mironowicz_ZZ_to_ZX_epsilon"]
        expected = cp.kron(z, np.cos(epsilon) * z + np.sin(epsilon) * x)

        np.testing.assert_allclose(cp.asnumpy(H.H_EE_matrix), cp.asnumpy(expected))
        self.assertEqual(H.bondsEE, [[1, 2], [2, 3]])
        self.assertEqual(H.J_EE, [0.04, 0.04])

    def test_mironowicz_rand_ee_zz_to_zx_sz_conservation_depends_on_epsilon(self):
        params = mironowicz_params("Mironowicz_rand_EE_ZZ_to_ZX")
        params["Mironowicz_H_E"] = "Z"
        params["Mironowicz_theta"] = np.pi / 2
        params["Mironowicz_ZZ_to_ZX_epsilon"] = 0.0

        conserved, reason = check_sz_conservation(params)
        self.assertTrue(conserved, reason)

        params["Mironowicz_ZZ_to_ZX_epsilon"] = 0.1
        conserved, reason = check_sz_conservation(params)
        self.assertFalse(conserved)
        self.assertIn("ZZ_to_ZX", reason)

    def test_paper_b_disjoint_source_and_target_field_masks(self):
        base = mironowicz_params("PaperB_staged")
        base.update(
            {
                "nqubits_E": 4,
                "H_SE_p": "Const",
                "PaperB_field_strength": 0.6,
            }
        )
        source_params = dict(base, PaperB_case="zx_gaussian_disjoint_sources")
        target_params = dict(base, PaperB_case="zx_gaussian_disjoint_targets")

        source = hamiltonian(source_params, seed=source_params["seed"])
        target = hamiltonian(target_params, seed=target_params["seed"])

        self.assertEqual(source.bondsEE, [[1, 2], [3, 4]])
        self.assertNotEqual(source.J_E[0], 0.0)
        self.assertEqual(source.J_E[1], 0.0)
        self.assertNotEqual(source.J_E[2], 0.0)
        self.assertEqual(source.J_E[3], 0.0)
        self.assertEqual(target.J_E[0], 0.0)
        self.assertNotEqual(target.J_E[1], 0.0)
        self.assertEqual(target.J_E[2], 0.0)
        self.assertNotEqual(target.J_E[3], 0.0)

    def test_component_views_separate_write_and_storage_terms(self):
        params = mironowicz_params("PaperB_staged")
        params.update(
            {
                "H_SE_p": "Const",
                "PaperB_case": "zx_gaussian_chain",
                "PaperB_field_strength": 0.6,
            }
        )
        H = hamiltonian(params, seed=params["seed"])

        write = H.component_view(include_SE=True, include_E=False, include_EE=False)
        store = H.component_view(include_SE=False, include_E=True, include_EE=True)

        self.assertTrue(any(value != 0 for value in write.J_SE))
        self.assertTrue(all(value == 0 for value in write.J_E))
        self.assertTrue(all(value == 0 for value in write.J_EE))
        self.assertTrue(all(value == 0 for value in store.J_SE))
        self.assertTrue(any(value != 0 for value in store.J_E))
        self.assertTrue(any(value != 0 for value in store.J_EE))


if __name__ == "__main__":
    unittest.main()
