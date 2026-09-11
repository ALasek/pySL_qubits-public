#!/usr/bin/env python3
"""Run the profile-controlled Paper-A simulation and figure suite."""

import argparse
import datetime
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.batch_utils import count_parameter_combinations  # noqa: E402


PROFILE_ENV = "PYSL_PAPER_A_PROFILE"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "paper_a"
JOBS = (
    "lambda_collapse",
    "uniform_field",
    "disorder_field",
    "stability_map",
)
CONFIGS = {name: f"batch_configs.paper_a_{name}" for name in JOBS}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def paper_a_profile(profile):
    previous = os.environ.get(PROFILE_ENV)
    os.environ[PROFILE_ENV] = profile
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PROFILE_ENV, None)
        else:
            os.environ[PROFILE_ENV] = previous


def load_job_config(job, profile):
    with paper_a_profile(profile):
        common = importlib.import_module("batch_configs.paper_a_common")
        importlib.reload(common)
        module = importlib.import_module(CONFIGS[job])
        module = importlib.reload(module)
        return module.get_config()


def batch_command(job, args):
    command = [
        sys.executable,
        "pySLbatch.py",
        "--config",
        CONFIGS[job],
        "--workers",
        str(args.workers),
        "--gpu-ids",
        args.gpu_ids,
    ]
    if args.manifest_only:
        command.append("--manifest-only")
    elif args.skip_existing:
        command.append("--skip-existing")
    if args.worker_stagger_seconds:
        command.extend(["--worker-stagger-seconds", str(args.worker_stagger_seconds)])
    return command


def analysis_commands(job, profile):
    config = load_job_config(job, profile)
    analysis = config["analysis"]
    base = [
        sys.executable,
        "pySL_batchAnalyze.py",
        "--subdir",
        config["subdir"],
        "--x",
        analysis["x"],
        "--Tsample",
        str(analysis["Tsample"]),
        "--setMinTime",
        str(analysis["setMinTime"]),
        "--holevo-delta",
        str(analysis["holevo_delta"]),
        "--holevo-quantile",
        str(analysis["holevo_quantile"]),
        "--no-pane",
    ]
    if analysis.get("y"):
        base.extend(["--y", analysis["y"]])

    if job == "lambda_collapse":
        return [
            base + ["--filter", f"nqubits_E={nqubits}", "--analysis-name", f"lambda_N{nqubits}"]
            for nqubits in config["sweep"]["nqubits_E"]
        ]
    if job == "stability_map":
        return [
            base
            + [
                "--filter",
                f"Mironowicz_theta={theta}",
                "--analysis-name",
                f"stability_theta_{float(theta):.6g}",
            ]
            for theta in config["sweep"]["Mironowicz_theta"]
        ]
    if job in {"uniform_field", "disorder_field"}:
        return [
            base
            + [
                "--filter",
                f"psi_bias={bias}",
                "--analysis-name",
                f"{job}_p_{float(bias):.6g}",
            ]
            for bias in config["sweep"]["psi_bias"]
        ]
    return [base]


def write_event(handle, payload):
    handle.write(json.dumps({"time": utc_now(), **payload}, sort_keys=True) + "\n")
    handle.flush()


def run_child(command, log_path, profile):
    env = os.environ.copy()
    env[PROFILE_ENV] = profile
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {PROFILE_ENV}={profile} {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def parse_jobs(raw):
    jobs = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(jobs) - set(JOBS))
    if unknown:
        raise ValueError(f"unknown Paper-A jobs: {', '.join(unknown)}")
    return jobs or list(JOBS)


def print_plan(jobs, args):
    print(f"profile: {args.profile}")
    print(f"workers: {args.workers}; gpu ids: {args.gpu_ids}; skip existing: {args.skip_existing}")
    for job in jobs:
        config = load_job_config(job, args.profile)
        count = count_parameter_combinations(config["sweep"])
        print(f"{job}: {count} combinations -> data/{config['subdir']}")
        if not args.figures_only:
            print(f"  run: {PROFILE_ENV}={args.profile} {' '.join(batch_command(job, args))}")
        if args.figures and not args.manifest_only:
            for command in analysis_commands(job, args.profile):
                print(f"  figures: {' '.join(command)}")


def run_phase(job, phase, command, log_dir, status, profile):
    log_path = log_dir / f"{job}_{phase}.log"
    write_event(status, {"event": f"{phase}_started", "job": job, "command": command, "log_path": str(log_path)})
    result = run_child(command, log_path, profile)
    event = f"{phase}_completed" if result == 0 else f"{phase}_failed"
    write_event(status, {"event": event, "job": job, "return_code": result, "log_path": str(log_path)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the profile-controlled Paper-A suite.")
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--jobs", default=",".join(JOBS), help="Comma-separated job names.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts, subdirectories, and commands without writing.")
    parser.add_argument("--manifest-only", action="store_true", help="Write each pySLbatch manifest without simulations.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true", default=True, help="Skip existing run results."
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Rerun all combinations.")
    parser.add_argument("--worker-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.worker_stagger_seconds < 0:
        parser.error("--worker-stagger-seconds must be non-negative")
    if args.figures_only and not args.figures:
        parser.error("--figures-only cannot be combined with --no-figures")
    if args.manifest_only and args.figures_only:
        parser.error("--manifest-only cannot be combined with --figures-only")
    try:
        jobs = parse_jobs(args.jobs)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print_plan(jobs, args)
        return 0

    log_dir = args.log_dir or DEFAULT_LOG_ROOT / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(log_dir / "orchestration_status.jsonl", "a", encoding="utf-8", buffering=1) as status:
        write_event(status, {"event": "suite_started", "profile": args.profile, "jobs": jobs})
        for job in jobs:
            if not args.figures_only:
                result = run_phase(job, "batch", batch_command(job, args), log_dir, status, args.profile)
                if result:
                    failures += 1
                    if not args.keep_going:
                        break
            if args.figures and not args.manifest_only:
                for index, command in enumerate(analysis_commands(job, args.profile), start=1):
                    result = run_phase(job, f"figures_{index}", command, log_dir, status, args.profile)
                    if result:
                        failures += 1
                        if not args.keep_going:
                            break
                if failures and not args.keep_going:
                    break
        write_event(status, {"event": "suite_finished", "failures": failures})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
