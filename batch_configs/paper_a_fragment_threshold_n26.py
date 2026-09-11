"""Optional N_E=26 extension of the Paper-A fragment-threshold checks."""

import math

from batch_configs.paper_a_common import build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FragmentThresholdN26"
params_default = build_params(
    T=36 if _final else 0.02,
    extra={
        "AverageOverRunsN": 8 if _final else 1,
        "nqubits_E": 26,
        "dT": 0.002,
        "printT": 36 if _final else 0.02,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "log_gpu_memory": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)
cases = [
    {
        "validation_case": f"lambda_{lambda_target:g}_N26",
        "validation_family": "finite_environment_fragment_threshold",
        "nqubits_E": 26,
        "lambda_target": lambda_target,
        "lambda_geometry": "p_half",
        "psi_bias": 0.5,
        "Mironowicz_theta": math.asin(math.sqrt(lambda_target)),
    }
    for lambda_target in (0.75, 1.0)
]
analysis = {
    "purpose": "optional N_E=26 finite-environment comparison for Paper-A Fig. 5(b)",
    "primary_time": 36.0 if _final else 0.02,
    "reference_environment_sizes": [8, 12, 16, 20, 24],
    "record_holevo_fraction": 0.9,
    "fragment_quantile": 0.1,
    "threshold_estimator": "first crossing of the run-averaged within-realization q10 Holevo curve",
    "claim_boundary": "finite-pool consistency check against an exactly N_E-independent iid benchmark",
    "smoke_scope": "both Hamiltonian angles, N_E=26, ten production-size steps, and fragments through m=13; four rather than 64 samples per size",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
