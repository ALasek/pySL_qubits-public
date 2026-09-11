import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("inherited,explicit,expected", [
    (None, "final", "final"),
    ("smoke", "final", "final"),
    ("final", "smoke", "smoke"),
    ("final", None, "final"),
])
def test_profile_resolved_before_config_import(inherited, explicit, expected):
    env = os.environ.copy()
    env.pop("PYSL_PAPER_A_PROFILE", None)
    if inherited:
        env["PYSL_PAPER_A_PROFILE"] = inherited
    code = """
import json, os, pySLbatch
def capture(**kwargs):
    print('CONFIG=' + json.dumps([kwargs['subdir'],
          kwargs['params_default']['AverageOverRunsN'],
          len(kwargs['cases']), os.environ['PYSL_PAPER_A_PROFILE']]))
pySLbatch.write_batch_manifest = capture
pySLbatch.main()
"""
    command = [sys.executable, "-c", code, "--config",
               "batch_configs.paper_a_low_field_refinement", "--manifest-only"]
    if explicit:
        command += ["--profile", explicit]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                            env=env, check=True, capture_output=True, text=True)
    record = next(line.removeprefix("CONFIG=") for line in result.stdout.splitlines()
                  if line.startswith("CONFIG="))
    assert json.loads(record) == [f"Paper_A_{expected}_LowFieldRefinement",
                                 24 if expected == "final" else 1, 8, expected]
