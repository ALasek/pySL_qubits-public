#!/usr/bin/env python3
"""
Run QD_summary replacement simulations from qdsummary_replacements/plot_rebuild_manifest.json.

Single-run JSONs are executed first. Batch configs are then executed sequentially,
with each batch using pySLbatch.py --workers 4 by default.
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.batch_utils import build_batch_tasks, load_batch_config
from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.run_store import RESULTS_FILENAME, compute_run_id
from src.input.param_expressions import resolve_parameter_expressions
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import validate_params

DEFAULT_MANIFEST = REPO_ROOT / "qdsummary_replacements" / "plot_rebuild_manifest.json"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "qdsummary_replacements"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def timestamp_for_path():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sanitize_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:120] or "job"


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ordered_jobs(manifest):
    singles = []
    batches = []
    seen = set()
    for entry in manifest["replacements"]:
        replacement_path = entry["replacement_path"]
        key = (entry["replacement_type"], replacement_path)
        if key in seen:
            continue
        seen.add(key)

        job = {
            "labels": entry.get("labels", []),
            "replacement_path": replacement_path,
            "replacement_type": entry["replacement_type"],
            "notes": entry.get("notes", ""),
        }
        if entry["replacement_type"] == "run_params_json":
            job["phase"] = "single"
            job["name"] = Path(replacement_path).stem
            singles.append(job)
        elif entry["replacement_type"] == "batch_config":
            job["phase"] = "batch"
            job["name"] = Path(replacement_path).stem
            batches.append(job)
        else:
            raise ValueError(f"Unknown replacement_type: {entry['replacement_type']!r}")
    return singles + batches


def single_params_path(job):
    path = REPO_ROOT / job["replacement_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def single_params_arg(path):
    run_params_root = REPO_ROOT / "run_params"
    return str(path.relative_to(run_params_root)).replace("\\", "/")


def single_result_path(job):
    path = single_params_path(job)
    params = json.loads(path.read_text(encoding="utf-8"))
    params = resolve_parameter_expressions(params)
    validate_params(params)
    stored_params = dict(params)
    stored_params.update(resolve_run_seed_plan_from_params(stored_params))
    run_id = compute_run_id(stored_params)
    return Path(get_data_dir(DEFAULT_RUN_SUBDIR)) / "runs" / run_id / RESULTS_FILENAME


def batch_config_ref(job):
    path = Path(job["replacement_path"])
    return "batch_configs." + path.stem


def batch_remaining(job, workers, gpu_ids):
    config = load_batch_config(batch_config_ref(job))
    plan = build_batch_tasks(
        params_default=config["params_default"],
        loops_dist=config["sweep"],
        subdir=config["subdir"],
        workers=workers,
        gpu_ids=gpu_ids,
        skip_existing=True,
    )
    return {
        "subdir": config["subdir"],
        "total": plan["total"],
        "remaining": len(plan["tasks"]),
        "skipped_existing": len(plan["skipped"]),
    }


def command_for_job(job, args):
    if job["phase"] == "single":
        params_path = single_params_path(job)
        return [
            sys.executable,
            "pySL.py",
            "--params",
            single_params_arg(params_path),
        ]

    cmd = [
        sys.executable,
        "pySLbatch.py",
        "--config",
        batch_config_ref(job),
        "--workers",
        str(args.workers),
        "--gpu-ids",
        args.gpu_ids,
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.worker_stagger_seconds:
        cmd.extend(["--worker-stagger-seconds", str(args.worker_stagger_seconds)])
    return cmd


def figure_command_for_job(job, args):
    if job["phase"] == "single":
        return [
            sys.executable,
            "scripts/plot_qdsummary_single_replacements.py",
            "--manifest",
            args.manifest,
            "--replacement-path",
            job["replacement_path"],
        ]
    config = load_batch_config(batch_config_ref(job))
    return [
        sys.executable,
        "pySL_batchAnalyze.py",
        "--subdir",
        config["subdir"],
        "--no-pane",
    ]


def write_event(handle, payload):
    handle.write(json.dumps({"time": utc_now(), **payload}, sort_keys=True) + "\n")
    handle.flush()


def run_child(command, log_path):
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def should_run_job(job, args):
    if not args.skip_existing:
        return True, {}
    if job["phase"] == "single":
        result_path = single_result_path(job)
        return not result_path.exists(), {"result_path": str(result_path)}
    status = batch_remaining(job, args.workers, args.gpu_ids)
    return status["remaining"] > 0, status


def print_plan(jobs, args):
    print(f"manifest: {args.manifest}")
    print(f"workers for batches: {args.workers}")
    print(f"gpu ids: {args.gpu_ids}")
    print(f"skip existing: {args.skip_existing}")
    print(f"figures: {args.figures}")
    print(f"figures only: {args.figures_only}")
    print(f"jobs: {len(jobs)}")
    for index, job in enumerate(jobs, start=1):
        command = " ".join(command_for_job(job, args))
        if args.figures_only:
            run, status = False, {}
            state = "figures-only"
        else:
            run, status = should_run_job(job, args)
            state = "run" if run else "skip"
        details = ""
        if job["phase"] == "batch" and status:
            details = (
                f" total={status['total']}"
                f" remaining={status['remaining']}"
                f" skipped_existing={status['skipped_existing']}"
            )
        print(f"{index:02d}. [{state}] {job['phase']} {job['name']}{details}")
        if not args.figures_only:
            print(f"    run: {command}")
        if args.figures:
            print(f"    figures: {' '.join(figure_command_for_job(job, args))}")


def main():
    parser = argparse.ArgumentParser(description="Run all QD_summary replacement simulations.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Replacement manifest JSON.")
    parser.add_argument("--workers", type=int, default=4, help="Workers for each batch run.")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids passed to pySLbatch.py.")
    parser.add_argument(
        "--worker-stagger-seconds",
        type=float,
        default=0.0,
        help="Initial worker launch stagger passed to pySLbatch.py.",
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip completed single runs and pass --skip-existing to batches.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Rerun all jobs even if stored results already exist.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "single", "batch"),
        default="all",
        help="Restrict execution to one phase.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan and exit.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed job.")
    parser.add_argument(
        "--figures",
        dest="figures",
        action="store_true",
        default=True,
        help="Generate figures after each completed or skipped-existing job.",
    )
    parser.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Run simulations only; do not run post-processing figure generation.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Do not run simulations; only generate figures from existing results.",
    )
    parser.add_argument("--log-dir", default=None, help="Override the run log directory.")
    args = parser.parse_args()

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.worker_stagger_seconds < 0:
        parser.error("--worker-stagger-seconds must be non-negative")
    if args.figures_only and not args.figures:
        parser.error("--figures-only cannot be combined with --no-figures")

    args.manifest = str(Path(args.manifest).resolve())
    manifest = load_manifest(args.manifest)
    jobs = ordered_jobs(manifest)
    if args.only != "all":
        jobs = [job for job in jobs if job["phase"] == args.only]

    if args.dry_run:
        print_plan(jobs, args)
        return 0

    log_dir = Path(args.log_dir) if args.log_dir else DEFAULT_LOG_ROOT / timestamp_for_path()
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "orchestration_status.jsonl"

    failures = 0
    with open(status_path, "a", encoding="utf-8", buffering=1) as status:
        write_event(
            status,
            {
                "event": "orchestration_started",
                "manifest": args.manifest,
                "workers": args.workers,
                "gpu_ids": args.gpu_ids,
                "skip_existing": args.skip_existing,
                "figures": args.figures,
                "figures_only": args.figures_only,
                "job_count": len(jobs),
                "log_dir": str(log_dir),
            },
        )

        for index, job in enumerate(jobs, start=1):
            if args.figures_only:
                run, skip_status = False, {"reason": "figures_only"}
            else:
                run, skip_status = should_run_job(job, args)
            log_path = log_dir / f"{index:03d}_{job['phase']}_{sanitize_name(job['name'])}.log"
            job_ready_for_figures = False
            if not run:
                print(f"[skip] {job['phase']} {job['name']}")
                write_event(
                    status,
                    {
                        "event": "job_skipped",
                        "index": index,
                        "job": job,
                        "skip_status": skip_status,
                    },
                )
                job_ready_for_figures = True
            else:
                command = command_for_job(job, args)
                print(f"[start] {index}/{len(jobs)} {job['phase']} {job['name']}")
                print(f"[log] {log_path}")
                started = time.time()
                write_event(
                    status,
                    {
                        "event": "job_started",
                        "index": index,
                        "job": job,
                        "command": command,
                        "log_path": str(log_path),
                        "preflight_status": skip_status,
                    },
                )
                try:
                    return_code = run_child(command, log_path)
                except KeyboardInterrupt:
                    write_event(
                        status,
                        {
                            "event": "job_interrupted",
                            "index": index,
                            "job": job,
                            "seconds": time.time() - started,
                            "log_path": str(log_path),
                        },
                    )
                    raise

                seconds = time.time() - started
                if return_code:
                    failures += 1
                    print(f"[failed] {job['phase']} {job['name']} rc={return_code}")
                    write_event(
                        status,
                        {
                            "event": "job_failed",
                            "index": index,
                            "job": job,
                            "return_code": return_code,
                            "seconds": seconds,
                            "log_path": str(log_path),
                        },
                    )
                    if not args.keep_going:
                        break
                else:
                    job_ready_for_figures = True
                    print(f"[done] {job['phase']} {job['name']} seconds={seconds:.1f}")
                    write_event(
                        status,
                        {
                            "event": "job_completed",
                            "index": index,
                            "job": job,
                            "return_code": return_code,
                            "seconds": seconds,
                            "log_path": str(log_path),
                        },
                    )

            if args.figures and job_ready_for_figures:
                figure_command = figure_command_for_job(job, args)
                figure_log_path = log_dir / f"{index:03d}_{job['phase']}_{sanitize_name(job['name'])}_figures.log"
                print(f"[figures] {job['phase']} {job['name']}")
                print(f"[log] {figure_log_path}")
                started = time.time()
                write_event(
                    status,
                    {
                        "event": "figures_started",
                        "index": index,
                        "job": job,
                        "command": figure_command,
                        "log_path": str(figure_log_path),
                    },
                )
                try:
                    return_code = run_child(figure_command, figure_log_path)
                except KeyboardInterrupt:
                    write_event(
                        status,
                        {
                            "event": "figures_interrupted",
                            "index": index,
                            "job": job,
                            "seconds": time.time() - started,
                            "log_path": str(figure_log_path),
                        },
                    )
                    raise
                seconds = time.time() - started
                if return_code:
                    failures += 1
                    print(f"[figures failed] {job['phase']} {job['name']} rc={return_code}")
                    write_event(
                        status,
                        {
                            "event": "figures_failed",
                            "index": index,
                            "job": job,
                            "return_code": return_code,
                            "seconds": seconds,
                            "log_path": str(figure_log_path),
                        },
                    )
                    if not args.keep_going:
                        break
                else:
                    print(f"[figures done] {job['phase']} {job['name']} seconds={seconds:.1f}")
                    write_event(
                        status,
                        {
                            "event": "figures_completed",
                            "index": index,
                            "job": job,
                            "return_code": return_code,
                            "seconds": seconds,
                            "log_path": str(figure_log_path),
                        },
                    )

        write_event(status, {"event": "orchestration_finished", "failures": failures})

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
