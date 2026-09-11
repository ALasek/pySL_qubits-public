#!/usr/bin/env python3
"""Run the staged write-store suites for candidate Paper B."""

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


PROFILE_ENV = "PYSL_PAPER_B_PROFILE"
CONFIG = "batch_configs.paper_b_focus"
MATCHED_CONFIG = "batch_configs.paper_b_matched_strength"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs" / "paper_b"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@contextmanager
def paper_b_profile(profile):
    previous = os.environ.get(PROFILE_ENV)
    os.environ[PROFILE_ENV] = profile
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PROFILE_ENV, None)
        else:
            os.environ[PROFILE_ENV] = previous


def load_config(profile):
    if profile == "matched":
        module = importlib.import_module(MATCHED_CONFIG)
        module = importlib.reload(module)
        return module.get_config()
    with paper_b_profile(profile):
        module = importlib.import_module(CONFIG)
        module = importlib.reload(module)
        return module.get_config()


def batch_command(args):
    config_module = MATCHED_CONFIG if args.profile == "matched" else CONFIG
    command = [
        sys.executable,
        "pySLbatch.py",
        "--config",
        config_module,
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


def analysis_commands(profile):
    config = load_config(profile)
    analysis = config["analysis"]
    commands = []
    for nqubits in config["sweep"]["nqubits_E"]:
        commands.append(
            [
                sys.executable,
                "pySL_batchAnalyze.py",
                "--subdir",
                config["subdir"],
                "--x",
                analysis["x"],
                "--y",
                analysis["y"],
                "--Tsample",
                str(config["params_default"]["T"]),
                "--setMinTime",
                str(analysis["setMinTime"]),
                "--holevo-delta",
                str(analysis["holevo_delta"]),
                "--holevo-quantile",
                str(analysis["holevo_quantile"]),
                "--filter",
                f"nqubits_E={nqubits}",
                "--analysis-name",
                f"N{nqubits}_final_late_window",
                "--no-pane",
                "--metrics",
                *analysis["metrics"],
            ]
        )
    return commands


def write_event(handle, payload):
    handle.write(json.dumps({"time": utc_now(), **payload}, sort_keys=True) + "\n")
    handle.flush()


def run_child(command, log_path, profile):
    env = os.environ.copy()
    profile_prefix = ""
    if profile != "matched":
        env[PROFILE_ENV] = profile
        profile_prefix = f"{PROFILE_ENV}={profile} "
    else:
        env.pop(PROFILE_ENV, None)
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ {profile_prefix}{' '.join(command)}\n")
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


def print_plan(args):
    config = load_config(args.profile)
    count = count_parameter_combinations(config["sweep"])
    print(f"Paper B profile: {args.profile}")
    print(f"combinations: {count} -> data/{config['subdir']}")
    print(f"workers: {args.workers}; gpu ids: {args.gpu_ids}; skip existing: {args.skip_existing}")
    if not args.figures_only:
        profile_prefix = "" if args.profile == "matched" else f"{PROFILE_ENV}={args.profile} "
        print(f"run: {profile_prefix}{' '.join(batch_command(args))}")
    if args.figures and not args.manifest_only:
        for command in analysis_commands(args.profile):
            print(f"figures: {' '.join(command)}")


def run_phase(phase, command, log_dir, status, profile):
    log_path = log_dir / f"{phase}.log"
    write_event(status, {"event": f"{phase}_started", "command": command, "log_path": str(log_path)})
    result = run_child(command, log_path, profile)
    event = f"{phase}_completed" if result == 0 else f"{phase}_failed"
    write_event(status, {"event": event, "return_code": result, "log_path": str(log_path)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a Paper-B staged write-store suite.")
    parser.add_argument("--profile", choices=("smoke", "focused", "matched"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--worker-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--figures-only", action="store_true")
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
    if args.dry_run:
        print_plan(args)
        return 0

    log_dir = args.log_dir or DEFAULT_LOG_ROOT / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(log_dir / "orchestration_status.jsonl", "a", encoding="utf-8", buffering=1) as status:
        write_event(status, {"event": "suite_started", "profile": args.profile})
        if not args.figures_only:
            failures += bool(run_phase("batch", batch_command(args), log_dir, status, args.profile))
        if not failures and args.figures and not args.manifest_only:
            for index, command in enumerate(analysis_commands(args.profile), start=1):
                if run_phase(f"figures_{index}", command, log_dir, status, args.profile):
                    failures += 1
                    break
        write_event(status, {"event": "suite_finished", "failures": failures})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
