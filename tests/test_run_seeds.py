import unittest
from unittest.mock import patch

from src.input.run_seeds import resolve_run_seed_plan, resolve_run_seed_plan_from_params


class RunSeedTests(unittest.TestCase):
    def test_single_fixed_seed_is_preserved(self):
        plan = resolve_run_seed_plan(1337, 1)

        self.assertEqual(plan["run_seeds"], [1337])
        self.assertEqual(plan["seed_base"], 1337)
        self.assertEqual(plan["seed_strategy"], "fixed")

    def test_multi_run_fixed_seed_is_deterministic_and_distinct(self):
        left = resolve_run_seed_plan(1337, 4)
        right = resolve_run_seed_plan(1337, 4)

        self.assertEqual(left["run_seeds"], right["run_seeds"])
        self.assertEqual(len(set(left["run_seeds"])), 4)

    def test_time_seed_records_concrete_seed(self):
        with patch("src.input.run_seeds.time.time_ns", return_value=123456789):
            plan = resolve_run_seed_plan("TIME", 2)

        self.assertEqual(plan["seed_input"], "TIME")
        self.assertEqual(plan["seed_base"], 123456789)
        self.assertEqual(len(plan["run_seeds"]), 2)

    def test_existing_run_seeds_are_reused(self):
        params = {
            "AverageOverRunsN": 2,
            "seed": "TIME",
            "seed_base": 10,
            "run_seeds": [11, 12],
        }

        plan = resolve_run_seed_plan_from_params(params)

        self.assertEqual(plan["seed_base"], 10)
        self.assertEqual(plan["run_seeds"], [11, 12])
        self.assertEqual(plan["seed_strategy"], "provided")


if __name__ == "__main__":
    unittest.main()
