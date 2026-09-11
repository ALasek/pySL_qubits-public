#!/usr/bin/env python3
"""
Run the remaining alpha2-sensitive QD_summary replacement batches.

This is a focused wrapper around scripts/run_qdsummary_replacements.py. It uses
qdsummary_replacements/remaining_bad_alpha2_manifest.json, which excludes the
single-run replacements already copied into QD_summary/Figs_replace and the
epsilon-only alpha2=0 sweeps.
"""

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "qdsummary_replacements" / "remaining_bad_alpha2_manifest.json"
RUNNER = REPO_ROOT / "scripts" / "run_qdsummary_replacements.py"


def main():
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest",
        str(MANIFEST),
        *sys.argv[1:],
    ]
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
