import unittest

from src.input.param_expressions import resolve_parameter_expressions, try_resolve_expression


class ParamExpressionTests(unittest.TestCase):
    def test_resolves_common_pi_expressions(self):
        self.assertAlmostEqual(try_resolve_expression("pi/2"), 1.5707963267948966)
        self.assertAlmostEqual(try_resolve_expression("3*pi/8"), 1.1780972450961724)
        self.assertAlmostEqual(try_resolve_expression("np.pi/4"), 0.7853981633974483)
        self.assertAlmostEqual(try_resolve_expression("math.pi"), 3.141592653589793)

    def test_leaves_plain_strings_untouched(self):
        self.assertEqual(try_resolve_expression("ZZ_rand"), "ZZ_rand")
        self.assertEqual(try_resolve_expression("TIME"), "TIME")
        self.assertEqual(try_resolve_expression("x+"), "x+")

    def test_resolves_nested_structures(self):
        params = {
            "Mironowicz_theta": "pi/2",
            "seed": "TIME",
            "nested": [
                "np.pi/8",
                {"value": "3*pi/4", "label": "ZZ_rand"},
            ],
        }

        resolved = resolve_parameter_expressions(params)

        self.assertAlmostEqual(resolved["Mironowicz_theta"], 1.5707963267948966)
        self.assertEqual(resolved["seed"], "TIME")
        self.assertAlmostEqual(resolved["nested"][0], 0.39269908169872414)
        self.assertAlmostEqual(resolved["nested"][1]["value"], 2.356194490192345)
        self.assertEqual(resolved["nested"][1]["label"], "ZZ_rand")


if __name__ == "__main__":
    unittest.main()
