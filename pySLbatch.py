"""
CLI README

Usage:
  python pySLbatch.py [--config CONFIG] [--combo-start N] [--workers N] [--manifest-only]

Arguments:
  --config CONFIG     Python module path or .py file path for a batch config.
                      Defaults to batch_configs.mironowicz_theta_sweep.
  --profile PROFILE   Paper-A profile: smoke or final; overrides the environment.
  --combo-start N     Skip parameter sets before zero-based index N, useful for
                      resuming a partially completed batch.
  --workers N         Number of concurrent worker processes. Defaults to 1.
  --gpu-ids IDS       Comma-separated physical GPU ids assigned round-robin to
                      worker tasks. Defaults to 0.
  --skip-existing     Skip runs whose v2 results.npz already exists.
  --worker-stagger-seconds S
                      Delay the first worker launches by staggered intervals.
  --manifest-only     Write data/<subdir>/batch_manifest.json and exit without
                      running simulations.
"""

import argparse
import os

from src.batch_utils import (
    count_parameter_sets,
    load_batch_config,
    parse_gpu_ids,
    run_parameter_sweep,
    run_parameter_sweep_parallel,
    write_batch_manifest,
)


DEFAULT_CONFIG = "batch_configs.mironowicz_theta_sweep"


def main():
    parser = argparse.ArgumentParser(description="Run a pySL parameter batch.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Python config module or .py path (default: batch_configs.mironowicz_theta_sweep)",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "final"),
        help="Paper-A profile; overrides PYSL_PAPER_A_PROFILE before loading the config",
    )
    parser.add_argument(
        "--combo-start",
        type=int,
        default=0,
        help="Skip combinations before this zero-based index.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the batch manifest without running simulations.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent worker processes (default: 1).",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated physical GPU ids assigned round-robin to workers (default: 0).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose data/<subdir>/runs/<run_id>/results.npz already exists.",
    )
    parser.add_argument(
        "--worker-stagger-seconds",
        type=float,
        default=0.0,
        help="Delay initial worker launches by staggered intervals to reduce allocation spikes.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.worker_stagger_seconds < 0:
        parser.error("--worker-stagger-seconds must be non-negative")
    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except ValueError as exc:
        parser.error(str(exc))

    if args.profile is not None:
        os.environ["PYSL_PAPER_A_PROFILE"] = args.profile
    config = load_batch_config(args.config)
    params_default = config["params_default"]
    sweep = config.get("sweep")
    cases = config.get("cases")
    subdir = config["subdir"]
    analysis = config.get("analysis", {})

    nruns = count_parameter_sets(loops_dist=sweep, cases=cases)
    print(f"{nruns} runs")
    print(f"subdir = {subdir}")
    print(f"config = {args.config}")
    print(f"workers = {args.workers}")
    print(f"gpu_ids = {gpu_ids}")
    print(f"skip_existing = {args.skip_existing}")

    if args.manifest_only:
        write_batch_manifest(
            params_default=params_default,
            loops_dist=sweep,
            subdir=subdir,
            analysis=analysis,
            status="manifest_only",
            cases=cases,
        )
        print("Wrote batch manifest")
        return

    if args.workers > 1 or args.skip_existing:
        summary = run_parameter_sweep_parallel(
            params_default=params_default,
            loops_dist=sweep,
            subdir=subdir,
            combo_start=args.combo_start,
            analysis=analysis,
            write_manifest=True,
            workers=args.workers,
            gpu_ids=gpu_ids,
            skip_existing=args.skip_existing,
            worker_stagger_seconds=args.worker_stagger_seconds,
            cases=cases,
        )
        print(f"Batch summary: {summary}")
        return

    from pySL import run_pySL
    run_parameter_sweep(
        params_default=params_default,
        loops_dist=sweep,
        runner=run_pySL,
        subdir=subdir,
        combo_start=args.combo_start,
        analysis=analysis,
        write_manifest=True,
        cases=cases,
    )


if __name__ == "__main__":
    main()
