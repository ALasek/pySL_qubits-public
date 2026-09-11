import unittest

import numpy as np

from src.input.wavefunction import _half_fragment_sizes, wavefunction


class FragmentSamplingTests(unittest.TestCase):
    def test_half_fragment_sizes_include_the_odd_environment_midpoint(self):
        self.assertEqual(list(_half_fragment_sizes(5)), [1, 2, 3])
        self.assertEqual(list(_half_fragment_sizes(6)), [1, 2, 3])

    def make_psi(self, *, reuse=True, count=5):
        psi = object.__new__(wavefunction)
        psi.nqubitsE = 8
        psi.seed = 1337
        psi.fragment_sample_count = count
        psi.fragment_reuse_samples = reuse
        psi._fragment_samples = {}
        return psi

    def test_reuses_deterministic_fragment_sets_within_a_realization(self):
        psi = self.make_psi()

        first = psi._fragment_combinations(3, 20)
        second = psi._fragment_combinations(3, 20)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)

    def test_fragment_sets_depend_on_run_seed(self):
        first = self.make_psi()._fragment_combinations(3, 20)
        second_psi = self.make_psi()
        second_psi.seed = 1338
        second = second_psi._fragment_combinations(3, 20)

        self.assertNotEqual(first, second)

    def test_stored_memberships_keep_realizations_separate(self):
        psi = self.make_psi(count=3)
        psi.store_fragment_samples = True
        psi.runN = 1
        psi.fragment_sample_members = np.full((9, 2, 3, 8), -1, dtype=np.int16)

        combinations = psi._fragment_combinations(2, 20)

        self.assertTrue(np.all(psi.fragment_sample_members[2, 0] == -1))
        for index, combination in enumerate(combinations):
            np.testing.assert_array_equal(
                psi.fragment_sample_members[2, 1, index, :2],
                combination,
            )


if __name__ == "__main__":
    unittest.main()
