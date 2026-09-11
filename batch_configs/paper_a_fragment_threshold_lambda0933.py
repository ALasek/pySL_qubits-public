"""Complete the red alignment curve at N_E=20,24 in Paper A Fig. 6(b)."""

import math

from batch_configs.paper_a_common import build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FragmentThresholdLambda0933"
params_default = build_params(
    T=36 if _final else 0.02,
    extra={
        "AverageOverRunsN": 8 if _final else 1,
        "nqubits_E": 24,
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
lambda_target = (2 + math.sqrt(3)) / 4
cases = [
    {
        "validation_case": f"lambda_{lambda_target:.9g}_N{n_environment}",
        "validation_family": "finite_environment_fragment_threshold",
        "nqubits_E": n_environment,
        "lambda_target": lambda_target,
        "lambda_geometry": "p_quarter",
        "psi_bias": 0.25,
        "Mironowicz_theta": math.pi / 4,
    }
    for n_environment in (20, 24)
]
analysis = {
    "purpose": "complete the red Lambda curve in Paper A Fig. 6(b)",
    "primary_time": 36.0 if _final else 0.02,
    "reference_batch": "Paper_A_final_FragmentThresholdFiniteN",
    "record_holevo_fraction": 0.9,
    "fragment_quantile": 0.1,
    "threshold_estimator": "within each realization: fragment q10, first threshold crossing, R=N_E/m; then average defined redundancies",
    "no_crossing_policy": "undefined if no crossing among m<=N_E/2; report conditional mean and crossing count",
    "uncertainty": "realization bootstrap with fragment selections held fixed",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
