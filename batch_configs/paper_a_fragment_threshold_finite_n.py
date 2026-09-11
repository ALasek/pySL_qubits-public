"""Fixed-time finite-environment threshold checks for Paper A."""

import math

from batch_configs.paper_a_common import build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FragmentThresholdFiniteN"
params_default = build_params(
    T=36 if _final else 0.02,
    extra={
        "AverageOverRunsN": 8 if _final else 1,
        "nqubits_E": 24,
        "dT": 0.002 if _final else 0.01,
        "printT": 36 if _final else 0.02,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)


def _case(n_environment, lambda_target):
    return {
        "validation_case": f"lambda_{lambda_target:g}_N{n_environment}",
        "validation_family": "finite_environment_fragment_threshold",
        "nqubits_E": n_environment,
        "lambda_target": lambda_target,
        "lambda_geometry": "p_half",
        "psi_bias": 0.5,
        "Mironowicz_theta": math.asin(math.sqrt(lambda_target)),
    }


cases = (
    [_case(n_environment, lambda_target) for n_environment in (20, 24) for lambda_target in (0.75, 1.0)]
    if _final
    else [_case(24, 1.0)]
)
analysis = {
    "purpose": "extend the finite-environment comparison in Paper-A Fig. 5(b)",
    "primary_time": 36.0 if _final else 0.02,
    "reference_environment_sizes": [8, 12, 16],
    "record_holevo_fraction": 0.9,
    "fragment_quantile": 0.1,
    "threshold_estimator": "first crossing of the run-averaged within-realization q10 Holevo curve",
    "claim_boundary": "finite-pool consistency check against an exactly N_E-independent iid benchmark",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
