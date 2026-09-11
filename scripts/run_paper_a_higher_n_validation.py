#!/usr/bin/env python3
"""Run the focused Paper-A higher-N validation batch."""

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


PROFILE_ENV = "PYSL_PAPER_A_HIGH_N_PROFILE"
CONFIG = "batch_configs.paper_a_higher_n_validation"


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


def load_config(profile):
    with paper_a_profile(profile):
        module = importlib.import_module(CONFIG)
        module = importlib.reload(module)
        return module.get_config()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")

    config = load_config(args.profile)
    count = count_parameter_sets(cases=config["cases"])
    command = [
        sys.executable,
        "pySLbatch.py",
        "--config",
        CONFIG,
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
    print(f"parameter cases: {count}")
    print(f"realizations per case: {config['params_default']['AverageOverRunsN']}")
    print(f"environment qubits: {config['params_default']['nqubits_E']}")
    print(f"subdir: data/{config['subdir']}")
    print("command:", " ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env[PROFILE_ENV] = args.profile
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
