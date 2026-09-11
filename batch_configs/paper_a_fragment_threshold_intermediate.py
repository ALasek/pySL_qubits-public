"""Intermediate-size state-vector points for Paper A's threshold comparison."""

import math

from batch_configs.paper_a_common import build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FragmentThresholdIntermediateN"
params_default = build_params(
    T=36 if _final else 0.02,
    extra={
        "AverageOverRunsN": 12 if _final else 1,
        "nqubits_E": 14,
        "dT": 0.002,
        "printT": 36 if _final else 0.02,
        # Match the N_E=8,12,16 LambdaCollapse fragment estimator.
        "fragment_sample_count": 32 if _final else 4,
        "fragment_quantiles": [0.1, 0.5],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)

_geometries = (
    (0.75, "p_half", 0.5, math.pi / 3),
    (0.5 + math.sqrt(3) / 4, "p_quarter", 0.25, math.pi / 4),
    (1.0, "p_half", 0.5, math.pi / 2),
)
cases = [
    {
        "validation_case": f"lambda_{target:.9g}_N{n_environment}",
        "validation_family": "finite_environment_fragment_threshold",
        "nqubits_E": n_environment,
        "lambda_target": target,
        "lambda_geometry": geometry,
        "psi_bias": p,
        "Mironowicz_theta": theta,
    }
    for n_environment in (10, 14)
    for target, geometry, p, theta in _geometries
]
analysis = {
    "purpose": "fill N_E=10,14 in the Paper A fragment-threshold comparison, currently Fig. 6(b)",
    "primary_time": 36.0 if _final else 0.02,
    "reference_batch": "Paper_A_final_LambdaCollapse",
    "reference_environment_sizes": [8, 12, 16],
    "record_holevo_fraction": 0.9,
    "fragment_quantile": 0.1,
    "threshold_estimator": "first crossing of the run-averaged within-realization q10 Holevo curve",
    "geometry_policy": "one representative of each Lambda from the original (p,theta) grid",
    "no_crossing_policy": "right-censored at m=N_E/2; do not discard noncrossing realizations",
    "claim_boundary": "finite-pool consistency check against an exactly N_E-independent iid benchmark",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
