import unittest
import numpy as np
from scipy.linalg import expm

from recalculate_exact import seeded, squared_overlaps, entropy


class ExactDynamicsTests(unittest.TestCase):
    def test_overlap_against_independent_matrix_exponentiation(self):
        x = np.array([[0, 1], [1, 0]])
        z = np.diag([1, -1])
        for theta, bias, field in [(0, .5, 0), (.73, .2, .18), (np.pi/2, .5, 3)]:
            p = dict(nqubits_E=5, H_SE_J=.1, Mironowicz_h0=field,
                     Mironowicz_alpha2=.3, Mironowicz_theta=theta, psi_bias=bias)
            seed, times = 3923535749, np.array([0, .1, 3.7, 60])
            predicted = squared_overlaps(p, seed, times)
            rg, rh = seeded(seed, 'mironowicz.J_SE'), seeded(seed, 'mironowicz.J_E')
            ket = np.sqrt([bias, 1-bias])
            expected = []
            for _ in range(5):
                g, h = rg.gauss(0, .1), field+rh.gauss(0, .3)
                h0 = h*z
                h1 = h*z-np.pi*g/2*(np.cos(theta)*x+np.sin(theta)*z)
                expected.append([abs(np.vdot(expm(-1j*h0*t)@ket, expm(-1j*h1*t)@ket))**2 for t in times])
            np.testing.assert_allclose(predicted, expected, atol=2e-13, rtol=2e-13)

    def test_empty_and_full_fragment_information_limits(self):
        be = np.array([0, .3, 1])
        hs = entropy(be)
        np.testing.assert_allclose(hs+entropy(np.ones(3))-entropy(be), 0)
        np.testing.assert_allclose(hs+entropy(be)-entropy(np.ones(3)), 2*hs)
        self.assertEqual(float(entropy(1)), 0)
        self.assertAlmostEqual(float(entropy(0)), np.log(2))


if __name__ == '__main__':
    unittest.main()
