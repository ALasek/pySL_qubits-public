import unittest
import os
import sys
import types
from unittest.mock import patch

from src.batch_utils import (
    _run_batch_task,
    build_batch_tasks,
    compute_stored_run_id,
    count_parameter_combinations,
    count_parameter_sets,
    iter_parameter_cases,
    iter_parameter_combinations,
    load_batch_config,
    parse_gpu_ids,
    run_parameter_sweep_parallel,
    run_parameter_sweep,
)
from src.export.run_store import compute_run_id
from src.input.run_seeds import resolve_run_seed_plan_from_params


class BatchUtilsTests(unittest.TestCase):
    def test_count_parameter_combinations(self):
        loops_dist = {"a": [1, 2], "b": ["x", "y", "z"]}
        self.assertEqual(count_parameter_combinations(loops_dist), 6)

    def test_explicit_cases_preserve_correlated_parameters(self):
        defaults = {"nested": {"items": [1]}, "fixed": True}
        cases = [{"p": 0.1, "theta": 0.2}, {"p": 0.3, "theta": 0.4}]

        generated = list(iter_parameter_cases(defaults, cases))
        generated[0]["nested"]["items"].append(2)

        self.assertEqual([(item["p"], item["theta"]) for item in generated], [(0.1, 0.2), (0.3, 0.4)])
        self.assertEqual(generated[1]["nested"]["items"], [1])
        self.assertEqual(count_parameter_sets(cases=cases), 2)

    def test_parameter_set_api_requires_exactly_one_mode(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            count_parameter_sets()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            count_parameter_sets(loops_dist={"x": [1]}, cases=[{"x": 1}])

    def test_combo_start_skips_prefix_but_executes_rest(self):
        params_default = {"base": True}
        loops_dist = {"a": [1, 2], "b": [10, 20]}
        calls = []

        def fake_runner(*, params, subdir):
            calls.append((params["a"], params["b"], subdir))

        total = run_parameter_sweep(
            params_default=params_default,
            loops_dist=loops_dist,
            runner=fake_runner,
            subdir="demo",
            combo_start=2,
        )

        self.assertEqual(total, 4)
        self.assertEqual(calls, [(2, 10, "demo"), (2, 20, "demo")])

    def test_parameter_combinations_deep_copy_nested_values(self):
        params_default = {"nested": {"items": [1, 2]}, "base": True}
        loops_dist = {"a": [1, 2]}

        combos = list(iter_parameter_combinations(params_default, loops_dist))
        combos[0]["nested"]["items"].append(3)

        self.assertEqual(params_default["nested"]["items"], [1, 2])
        self.assertEqual(combos[1]["nested"]["items"], [1, 2])

    def test_load_batch_config_accepts_bare_batch_config_name(self):
        config = load_batch_config("mironowicz_theta_sweep")

        self.assertEqual(config["subdir"], "Mironowicz_rand_ThetaSweep")
        self.assertIn("Mironowicz_theta", config["sweep"])

    def test_parse_gpu_ids(self):
        self.assertEqual(parse_gpu_ids("0,2"), [0, 2])
        self.assertEqual(parse_gpu_ids([1]), [1])

        with self.assertRaisesRegex(ValueError, "at least one"):
            parse_gpu_ids("")

    def test_build_batch_tasks_assigns_gpus_and_staggers_initial_workers(self):
        plan = build_batch_tasks(
            params_default={"base": True},
            loops_dist={"a": [1, 2, 3, 4]},
            subdir="demo",
            gpu_ids=[0, 1],
            workers=3,
            worker_stagger_seconds=5,
        )

        self.assertEqual(plan["total"], 4)
        self.assertEqual([task["gpu_id"] for task in plan["tasks"]], [0, 1, 0, 1])
        self.assertEqual([task["stagger_delay"] for task in plan["tasks"]], [0, 5, 10, 0.0])

    def test_build_batch_tasks_accepts_explicit_cases(self):
        plan = build_batch_tasks(
            params_default={"fixed": True},
            cases=[{"p": 0.1, "theta": 0.2}, {"p": 0.3, "theta": 0.4}],
            gpu_ids=[0, 2],
            workers=2,
        )

        self.assertEqual(plan["total"], 2)
        self.assertEqual([task["gpu_id"] for task in plan["tasks"]], [0, 2])
        self.assertEqual(
            [(task["params"]["p"], task["params"]["theta"]) for task in plan["tasks"]],
            [(0.1, 0.2), (0.3, 0.4)],
        )

    def test_build_batch_tasks_skips_existing_results(self):
        with patch("src.batch_utils.run_result_exists", side_effect=[True, False]):
            plan = build_batch_tasks(
                params_default={},
                loops_dist={"a": [1, 2]},
                subdir="demo",
                skip_existing=True,
            )

        self.assertEqual(len(plan["skipped"]), 1)
        self.assertEqual(len(plan["tasks"]), 1)
        self.assertEqual(plan["tasks"][0]["params"]["a"], 2)

    def test_compute_stored_run_id_resolves_averaged_run_seeds(self):
        params = {"AverageOverRunsN": 3, "seed": 1337}
        saved_params = dict(params)
        saved_params.update(resolve_run_seed_plan_from_params(params))

        self.assertEqual(compute_stored_run_id(params), compute_run_id(saved_params))

    def test_compute_stored_run_id_keeps_time_seed_stable(self):
        params = {"AverageOverRunsN": 3, "seed": "TIME"}

        self.assertEqual(compute_stored_run_id(params), compute_run_id(params))

    def test_parallel_sweep_writes_status_with_patched_worker(self):
        statuses = []

        def fake_status(subdir, row, filename="batch_status.jsonl"):
            statuses.append((subdir, row["status"], row["combo_index"], filename))

        def fake_worker(task):
            return {
                "combo_index": task["combo_index"],
                "run_id": task["run_id"],
                "gpu_id": task["gpu_id"],
                "status": "completed",
                "seconds": 0.1,
            }

        with patch("src.batch_utils._run_batch_task", side_effect=fake_worker), \
                patch("src.batch_utils.append_batch_status", side_effect=fake_status):
            summary = run_parameter_sweep_parallel(
                params_default={},
                loops_dist={"a": [1, 2]},
                subdir="demo",
                workers=1,
                gpu_ids=[0],
                status_filename="unit_status.jsonl",
            )

        self.assertEqual(summary["completed"], 2)
        self.assertEqual(statuses, [
            ("demo", "completed", 0, "unit_status.jsonl"),
            ("demo", "completed", 1, "unit_status.jsonl"),
        ])

    def test_run_batch_task_sets_cuda_env_and_cleans_up(self):
        calls = []

        def fake_run_pySL(*, params, subdir):
            calls.append((params, subdir, os.environ.get("CUDA_VISIBLE_DEVICES")))

        fake_pysl = types.SimpleNamespace(run_pySL=fake_run_pySL)
        task = {
            "combo_index": 0,
            "run_id": "abc",
            "gpu_id": 2,
            "params": {"x": 1},
            "subdir": "demo",
            "stagger_delay": 0,
        }

        with patch.dict(sys.modules, {"pySL": fake_pysl}), \
                patch.dict(os.environ, {}, clear=False), \
                patch("src.batch_utils.cleanup_cupy_memory") as cleanup:
            result = _run_batch_task(task)

        self.assertEqual(calls, [({"x": 1}, "demo", "2")])
        self.assertEqual(result["status"], "completed")
        cleanup.assert_called_once_with("batch worker cleanup", include_device=True)

    def test_run_batch_task_cleans_up_after_failure(self):
        def fake_run_pySL(*, params, subdir):  # noqa: ARG001
            raise RuntimeError("boom")

        fake_pysl = types.SimpleNamespace(run_pySL=fake_run_pySL)
        task = {
            "combo_index": 0,
            "run_id": "abc",
            "gpu_id": 0,
            "params": {},
            "subdir": "demo",
            "stagger_delay": 0,
        }

        with patch.dict(sys.modules, {"pySL": fake_pysl}), \
                patch("src.batch_utils.cleanup_cupy_memory") as cleanup:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                _run_batch_task(task)

        cleanup.assert_called_once_with("batch worker cleanup", include_device=True)


if __name__ == "__main__":
    unittest.main()
