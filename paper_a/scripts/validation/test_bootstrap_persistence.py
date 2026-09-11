import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("bootstrap_persistence.py")
SPEC = importlib.util.spec_from_file_location("bootstrap_persistence", MODULE_PATH)
bootstrap_persistence = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bootstrap_persistence)


def test_constant_fragment_trajectories_have_exact_bootstrap_result():
    trajectories = [np.full((3, 10), 0.95 * np.log(2.0)) for _ in range(4)]

    result = bootstrap_persistence._bootstrap(trajectories, draws=80, seed=11)

    assert result["central_mean"] == 0.95
    assert result["original_success_count_ge_0_9"] == 4
    assert result["realization_bootstrap_ci_low"] == 0.95
    assert result["realization_bootstrap_ci_high"] == 0.95
    assert result["nested_bootstrap_ci_low"] == 0.95
    assert result["nested_bootstrap_ci_high"] == 0.95


def test_persistence_quantiles_each_time_before_taking_the_minimum():
    bits = np.ones((2, 10))
    bits[0, 0] = 0.0
    bits[1, 1] = 0.0

    persistence = bootstrap_persistence._persistence_minima(bits * np.log(2.0))
    quantile_of_fragment_minima = np.quantile(np.min(bits, axis=0), 0.1)

    assert persistence == 0.9
    assert quantile_of_fragment_minima == 0.0


def test_fragment_stage_adds_uncertainty_for_multiple_realizations():
    trajectories = []
    for _ in range(4):
        bits = np.ones((3, 10))
        bits[:, 0] = 0.0
        trajectories.append(bits * np.log(2.0))

    result = bootstrap_persistence._bootstrap(trajectories, draws=400, seed=28)

    assert result["central_mean"] == 0.9
    assert result["realization_bootstrap_ci_width"] == 0.0
    assert result["nested_bootstrap_ci_width"] > 0.0
