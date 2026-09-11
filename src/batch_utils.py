import copy
import concurrent.futures
import datetime
import importlib
import importlib.util
import itertools
import json
import os
import subprocess
import time
import traceback

from src.export.path_utils import get_data_dir
from src.export.run_store import RESULTS_FILENAME, compute_run_id
from src.gpu_memory import cleanup_cupy_memory
from src.input.run_seeds import resolve_run_seed_plan_from_params


MANIFEST_FILENAME = "batch_manifest.json"
BATCH_STATUS_FILENAME = "batch_status.jsonl"
MANIFEST_SCHEMA_VERSION = 2


def _as_sequence(value):
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _jsonable_value(value):
    if hasattr(value, "item"):
        return _jsonable_value(value.item())
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return value


def normalize_sweep_values(loops_dist):
    return {
        key: [_jsonable_value(value) for value in _as_sequence(loops_dist[key])]
        for key in loops_dist
    }


def build_batch_manifest(
    params_default,
    loops_dist=None,
    subdir="",
    analysis=None,
    status="created",
    cases=None,
):
    if (loops_dist is None) == (cases is None):
        raise ValueError("define exactly one of loops_dist or cases")
    sweep = normalize_sweep_values(loops_dist) if loops_dist is not None else {}
    normalized_cases = normalize_cases(cases) if cases is not None else None
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,
        "subdir": subdir,
        "combo_count": len(normalized_cases) if normalized_cases is not None else count_parameter_combinations(sweep),
        "sweep": sweep,
        "params_default": _jsonable_value(params_default),
        "analysis": _jsonable_value(analysis or {}),
        "source_provenance": _git_provenance(),
    }
    if normalized_cases is None:
        manifest["sweep_keys"] = list(sweep.keys())
    else:
        manifest["case_keys"] = sorted({key for case in normalized_cases for key in case})
        manifest["cases"] = normalized_cases
    return manifest


def normalize_cases(cases):
    return [_jsonable_value(dict(case)) for case in cases]


def _git_provenance():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git_commit": commit, "git_dirty": dirty}


def _params_for_stored_run_id(params):
    if params.get("seed") == "TIME" and "run_seeds" not in params:
        return params
    try:
        seed_plan = resolve_run_seed_plan_from_params(params)
    except (KeyError, TypeError, ValueError):
        return params
    resolved = copy.deepcopy(params)
    resolved.update(seed_plan)
    return resolved


def compute_stored_run_id(params):
    return compute_run_id(_params_for_stored_run_id(params))


def get_manifest_path(subdir=""):
    return os.path.join(get_data_dir(subdir), MANIFEST_FILENAME)


def write_batch_manifest(
    params_default,
    loops_dist=None,
    subdir="",
    analysis=None,
    status="created",
    cases=None,
):
    manifest = build_batch_manifest(
        params_default=params_default,
        loops_dist=loops_dist,
        subdir=subdir,
        analysis=analysis,
        status=status,
        cases=cases,
    )
    out_dir = get_data_dir(subdir)
    os.makedirs(out_dir, exist_ok=True)
    with open(get_manifest_path(subdir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_batch_manifest(subdir=""):
    with open(get_manifest_path(subdir), "r", encoding="utf-8") as f:
        return json.load(f)


def load_batch_config(config_ref):
    if config_ref.endswith(".py") or os.path.sep in config_ref or "/" in config_ref:
        path = os.path.abspath(config_ref)
        if not os.path.exists(path) and os.path.dirname(config_ref) == "":
            path = os.path.abspath(os.path.join("batch_configs", config_ref))
        spec = importlib.util.spec_from_file_location("batch_config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(config_ref)
        except ModuleNotFoundError as exc:
            if "." in config_ref or exc.name != config_ref:
                raise
            module = importlib.import_module(f"batch_configs.{config_ref}")

    if hasattr(module, "get_config"):
        config = module.get_config()
    else:
        config = {
            "subdir": module.subdir,
            "params_default": module.params_default,
            "sweep": getattr(module, "sweep", getattr(module, "loopsDist", None)),
            "cases": getattr(module, "cases", None),
            "analysis": getattr(module, "analysis", {}),
        }

    has_sweep = config.get("sweep") is not None
    has_cases = config.get("cases") is not None
    if has_sweep == has_cases:
        raise ValueError(f"batch config '{config_ref}' must define exactly one of sweep/loopsDist or cases")
    return config


def iter_parameter_combinations(params_default, loops_dist):
    keys = list(loops_dist.keys())
    value_lists = [_as_sequence(loops_dist[key]) for key in keys]
    for combo in itertools.product(*value_lists):
        updated = copy.deepcopy(params_default)
        for key, value in zip(keys, combo):
            updated[key] = value
        yield updated


def iter_parameter_cases(params_default, cases):
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError(f"batch cases must be dictionaries, got {type(case).__name__}")
        updated = copy.deepcopy(params_default)
        updated.update(copy.deepcopy(case))
        yield updated


def iter_parameter_sets(params_default, loops_dist=None, cases=None):
    if (loops_dist is None) == (cases is None):
        raise ValueError("define exactly one of loops_dist or cases")
    if cases is not None:
        yield from iter_parameter_cases(params_default, cases)
    else:
        yield from iter_parameter_combinations(params_default, loops_dist)


def count_parameter_combinations(loops_dist):
    total = 1
    for key in loops_dist:
        total *= len(_as_sequence(loops_dist[key]))
    return total


def count_parameter_sets(loops_dist=None, cases=None):
    if (loops_dist is None) == (cases is None):
        raise ValueError("define exactly one of loops_dist or cases")
    return len(cases) if cases is not None else count_parameter_combinations(loops_dist)


def _run_results_path(params, subdir=""):
    run_id = compute_stored_run_id(params)
    return os.path.join(get_data_dir(subdir), "runs", run_id, RESULTS_FILENAME)


def run_result_exists(params, subdir=""):
    return os.path.exists(_run_results_path(params, subdir=subdir))


def _batch_status_path(subdir="", filename=BATCH_STATUS_FILENAME):
    return os.path.join(get_data_dir(subdir), filename)


def append_batch_status(subdir, row, filename=BATCH_STATUS_FILENAME):
    out_dir = get_data_dir(subdir)
    os.makedirs(out_dir, exist_ok=True)
    payload = _jsonable_value({
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **row,
    })
    with open(_batch_status_path(subdir, filename=filename), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_gpu_ids(value):
    if value is None:
        return [0]
    if isinstance(value, (list, tuple)):
        ids = [int(item) for item in value]
    else:
        ids = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not ids:
        raise ValueError("gpu_ids must contain at least one GPU id")
    if any(gpu_id < 0 for gpu_id in ids):
        raise ValueError(f"gpu_ids must be non-negative, got {ids}")
    return ids


def build_batch_tasks(
    params_default,
    loops_dist=None,
    subdir="",
    combo_start=0,
    gpu_ids=None,
    workers=1,
    skip_existing=False,
    worker_stagger_seconds=0.0,
    cases=None,
):
    if combo_start < 0:
        raise ValueError(f"combo_start must be non-negative, got {combo_start}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if worker_stagger_seconds < 0:
        raise ValueError(f"worker_stagger_seconds must be non-negative, got {worker_stagger_seconds}")

    gpu_ids = parse_gpu_ids(gpu_ids)
    tasks = []
    skipped = []
    total = 0
    submitted = 0

    for combo_index, params in enumerate(iter_parameter_sets(params_default, loops_dist=loops_dist, cases=cases)):
        total += 1
        if combo_index < combo_start:
            continue
        run_id = compute_stored_run_id(params)
        if skip_existing and run_result_exists(params, subdir=subdir):
            skipped.append({
                "combo_index": combo_index,
                "run_id": run_id,
                "status": "skipped_existing",
            })
            continue
        task = {
            "combo_index": combo_index,
            "params": params,
            "subdir": subdir,
            "run_id": run_id,
            "gpu_id": gpu_ids[submitted % len(gpu_ids)],
            "stagger_delay": (
                worker_stagger_seconds * (submitted % workers)
                if submitted < workers
                else 0.0
            ),
        }
        tasks.append(task)
        submitted += 1

    return {
        "total": total,
        "tasks": tasks,
        "skipped": skipped,
    }


def _run_batch_task(task):
    gpu_id = task.get("gpu_id")
    if gpu_id is not None:
        os.environ["CUDA_DEVICE"] = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    stagger_delay = float(task.get("stagger_delay", 0.0))
    if stagger_delay > 0:
        time.sleep(stagger_delay)

    t0 = time.time()
    try:
        from pySL import run_pySL

        run_pySL(params=task["params"], subdir=task["subdir"])
        return {
            "combo_index": task["combo_index"],
            "run_id": task["run_id"],
            "gpu_id": gpu_id,
            "status": "completed",
            "seconds": time.time() - t0,
        }
    finally:
        cleanup_cupy_memory("batch worker cleanup", include_device=True)


def _make_process_pool(workers):
    try:
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            max_tasks_per_child=1,
        )
    except TypeError:
        return concurrent.futures.ProcessPoolExecutor(max_workers=workers)


def _terminate_executor_processes(executor, timeout=5.0):
    processes = getattr(executor, "_processes", None)
    if not processes:
        return

    for process in list(processes.values()):
        if process is not None and process.is_alive():
            process.terminate()
    for process in list(processes.values()):
        if process is not None:
            process.join(timeout=timeout)


def run_parameter_sweep(
    params_default,
    loops_dist,
    runner,
    subdir="",
    combo_start=0,
    analysis=None,
    write_manifest=False,
    cases=None,
):
    if combo_start < 0:
        raise ValueError(f"combo_start must be non-negative, got {combo_start}")

    if write_manifest:
        write_batch_manifest(
            params_default=params_default,
            loops_dist=loops_dist,
            subdir=subdir,
            analysis=analysis,
            status="started",
            cases=cases,
        )

    combo_count = 0
    for updated_params in iter_parameter_sets(params_default, loops_dist=loops_dist, cases=cases):
        if combo_count >= combo_start:
            runner(params=updated_params, subdir=subdir)
        combo_count += 1

    if write_manifest:
        write_batch_manifest(
            params_default=params_default,
            loops_dist=loops_dist,
            subdir=subdir,
            analysis=analysis,
            status="completed",
            cases=cases,
        )

    return combo_count


def run_parameter_sweep_parallel(
    params_default,
    loops_dist,
    subdir="",
    combo_start=0,
    analysis=None,
    write_manifest=False,
    workers=2,
    gpu_ids=None,
    skip_existing=False,
    worker_stagger_seconds=0.0,
    status_filename=BATCH_STATUS_FILENAME,
    cases=None,
):
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")

    if write_manifest:
        write_batch_manifest(
            params_default=params_default,
            loops_dist=loops_dist,
            subdir=subdir,
            analysis=analysis,
            status="started",
            cases=cases,
        )

    plan = build_batch_tasks(
        params_default=params_default,
        loops_dist=loops_dist,
        subdir=subdir,
        combo_start=combo_start,
        gpu_ids=gpu_ids,
        workers=workers,
        skip_existing=skip_existing,
        worker_stagger_seconds=worker_stagger_seconds,
        cases=cases,
    )

    for row in plan["skipped"]:
        append_batch_status(subdir, row, filename=status_filename)

    summary = {
        "total": plan["total"],
        "submitted": len(plan["tasks"]),
        "skipped": len(plan["skipped"]),
        "completed": 0,
        "failed": 0,
    }

    if workers == 1:
        for task in plan["tasks"]:
            try:
                result = _run_batch_task(task)
            except KeyboardInterrupt:
                append_batch_status(
                    subdir,
                    {
                        "combo_index": task["combo_index"],
                        "run_id": task["run_id"],
                        "gpu_id": task["gpu_id"],
                        "status": "interrupted",
                    },
                    filename=status_filename,
                )
                if write_manifest:
                    write_batch_manifest(
                        params_default=params_default,
                        loops_dist=loops_dist,
                        subdir=subdir,
                        analysis=analysis,
                        status="interrupted",
                        cases=cases,
                    )
                raise
            except Exception as exc:
                summary["failed"] += 1
                append_batch_status(
                    subdir,
                    {
                        "combo_index": task["combo_index"],
                        "run_id": task["run_id"],
                        "gpu_id": task["gpu_id"],
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    filename=status_filename,
                )
            else:
                summary["completed"] += 1
                append_batch_status(subdir, result, filename=status_filename)
    else:
        interrupted = False
        executor = _make_process_pool(workers)
        futures = {}
        try:
            futures = {executor.submit(_run_batch_task, task): task for task in plan["tasks"]}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    summary["failed"] += 1
                    append_batch_status(
                        subdir,
                        {
                            "combo_index": task["combo_index"],
                            "run_id": task["run_id"],
                            "gpu_id": task["gpu_id"],
                            "status": "failed",
                            "error": repr(exc),
                            "traceback": "".join(
                                traceback.format_exception(type(exc), exc, exc.__traceback__)
                            ),
                        },
                        filename=status_filename,
                    )
                else:
                    summary["completed"] += 1
                    append_batch_status(subdir, result, filename=status_filename)
        except KeyboardInterrupt:
            interrupted = True
            for future, task in futures.items():
                if future.done():
                    continue
                future.cancel()
                append_batch_status(
                    subdir,
                    {
                        "combo_index": task["combo_index"],
                        "run_id": task["run_id"],
                        "gpu_id": task["gpu_id"],
                        "status": "interrupted",
                    },
                    filename=status_filename,
                )
            _terminate_executor_processes(executor)
            if write_manifest:
                write_batch_manifest(
                    params_default=params_default,
                    loops_dist=loops_dist,
                    subdir=subdir,
                    analysis=analysis,
                    status="interrupted",
                    cases=cases,
                )
            raise
        finally:
            try:
                executor.shutdown(wait=not interrupted, cancel_futures=interrupted)
            except TypeError:
                executor.shutdown(wait=not interrupted)

    final_status = "failed" if summary["failed"] else "completed"
    if write_manifest:
        write_batch_manifest(
            params_default=params_default,
            loops_dist=loops_dist,
            subdir=subdir,
            analysis=analysis,
            status=final_status,
            cases=cases,
        )

    if summary["failed"]:
        raise RuntimeError(
            f"{summary['failed']} batch run(s) failed; see "
            f"{_batch_status_path(subdir, filename=status_filename)}"
        )

    return summary
