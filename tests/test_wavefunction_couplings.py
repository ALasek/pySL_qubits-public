import unittest

from src.input.wavefunction import _include_coupling


class WavefunctionCouplingTests(unittest.TestCase):
    def test_negative_self_field_couplings_are_included(self):
        self.assertTrue(_include_coupling(-0.25))
        self.assertTrue(_include_coupling(0.25))
        self.assertFalse(_include_coupling(0.0))


if __name__ == "__main__":
    unittest.main()
