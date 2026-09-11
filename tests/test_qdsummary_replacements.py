import unittest

from batch_configs.qdsummary_replace_fig_ZX_alpha2_HEE_sweep import get_config


class QDSummaryReplacementTests(unittest.TestCase):
    def test_zx_replacement_uses_pure_zx_builder_and_fresh_subdir(self):
        config = get_config()

        self.assertEqual(config["params_default"]["H_SE_Special"], "Mironowicz_rand_EE_ZX")
        self.assertEqual(config["subdir"], "qdsummary_replace_fig_ZX_alpha2_HEE_pureZX_randfix_V2")


if __name__ == "__main__":
    unittest.main()
