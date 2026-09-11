#!/usr/bin/env python3
"""Run the fixed-time finite-environment threshold checks for Paper A."""

import argparse
import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.batch_utils import count_parameter_sets  # noqa: E402


PROFILE_ENV = "PYSL_PAPER_A_PROFILE"
CONFIG = "batch_configs.paper_a_fragment_threshold_finite_n"
CONFIG_N26 = "batch_configs.paper_a_fragment_threshold_n26"
CONFIG_INTERMEDIATE = "batch_configs.paper_a_fragment_threshold_intermediate"


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


def _config_module_name(*, n26=False, intermediate=False):
    if n26 and intermediate:
        raise ValueError("--n26 and --intermediate select separate batches")
    if intermediate:
        return CONFIG_INTERMEDIATE
    return CONFIG_N26 if n26 else CONFIG


def load_config(profile, *, n26=False, intermediate=False):
    with paper_a_profile(profile):
        module = importlib.import_module(_config_module_name(n26=n26, intermediate=intermediate))
        module = importlib.reload(module)
        return module.get_config()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--n26", action="store_true", help="select the separate optional N_E=26 batch")
    selection.add_argument("--intermediate", action="store_true", help="select only the N_E=10,14 fill-in batch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")

    config = load_config(args.profile, n26=args.n26, intermediate=args.intermediate)
    command = [
        sys.executable,
        "pySLbatch.py",
        "--config",
        _config_module_name(n26=args.n26, intermediate=args.intermediate),
        "--workers",
        str(args.workers),
        "--gpu-ids",
        args.gpu_ids,
    ]
    if args.manifest_only:
        command.append("--manifest-only")
    elif args.skip_existing:
        command.append("--skip-existing")

    print(f"profile: {args.profile}")
    print(f"parameter cases: {count_parameter_sets(cases=config['cases'])}")
    print(f"realizations per case: {config['params_default']['AverageOverRunsN']}")
    print("environment qubits:", sorted({case["nqubits_E"] for case in config["cases"]}))
    print("lambda targets:", sorted({case["lambda_target"] for case in config["cases"]}))
    if args.n26:
        print("N_E=26 means 27 total qubits. Run the smoke check alone on the GPU before production.")
    if args.intermediate:
        print("N_E=10,14 means 11,15 total qubits; only t=0 and the endpoint are recorded.")
        print("A curve that does not cross 0.9 by m=N_E/2 remains censored, not a fitted point.")
    print(f"subdir: data/{config['subdir']}")
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env[PROFILE_ENV] = args.profile
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
